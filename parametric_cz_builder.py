import math
import numpy as np
from typing import Union, List

from qiskit.pulse import Play, SamplePulse, ShiftPhase, Schedule, Waveform, ControlChannel, DriveChannel, GaussianSquare, Drag, Gaussian
from qiskit.circuit import Gate, Qubit, QuantumCircuit
from qiskit.circuit.library import CPhaseGate
from qiskit import QiskitError
from qiskit.providers import basebackend


class ParametericCZBuilder:

    def __init__(self, backend: basebackend):
        """
        Initializes a Parameterized Controlled-Z gate builder.
        Args:
            backend: Backend for which to construct the gates.
        """
        self._inst_map = backend.defaults().instruction_schedule_map
        self._config = backend.configuration()

    def basis_gates(self) -> list:
        """
        Returns:
            basis_gates: A list of basis gates in the backend.
        """
        return self._config.basis_gates

    def inst_map(self):
        """
        Returns:
            inst_map: the instruction schedule map.
        """
        return self._inst_map

    @staticmethod
    def rescale_amp(instruction: Play, theta: float) -> Union[Play, None]:
        """
        Rescale the amplitude of a sample pulse.
        The samples are scaled linearly so that theta = np.pi/2 has no effect.

        Args:
            instruction: The instruction from which to create a new scaled instruction.
            theta: The angle that controls the scaling.
        """

        scale = theta / (np.pi / 2.)

        if isinstance(instruction.pulse, Drag):
            drag = instruction.pulse
            return Play(Drag(duration=drag.duration, amp=drag.amp*scale, sigma=drag.sigma, beta=drag.beta),
                        instruction.channel)

        if isinstance(instruction.pulse, Gaussian):
            gaus = instruction.pulse
            return Play(Drag(duration=gaus.duration, amp=gaus.amp*scale, sigma=gaus.sigma, beta=gaus.beta),
                        instruction.channel)

    @staticmethod
    def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int = 16,
                        phase_delta: float = 0, amp_increase: float = 0.) -> Play:
        """
        Args:
            instruction: The instruction from which to create a new shortened or lengthened instruction.
            theta: desired angle, pi/2 is assumed to be the angle that
                the schedule with name 'name' in 'sched' implements.
            sample_mult: All pulses must be a multiple of sample_mult.
            phase_delta: Multiplies the pulse samples by np.exp(1.0j*phase_delta).
            amp_increase: Multiplies the amplitude of the samples by (1. + amp_increase).
        """
        if not isinstance(instruction.pulse, GaussianSquare):
            raise QiskitError('Parameteric builder only stretches/compresses '
                              'GaussianSquare.')

        amp = instruction.pulse.amp
        width = instruction.pulse.width
        sigma = instruction.pulse.sigma
        n_sigmas = (instruction.pulse.duration - width) / sigma

        # The error function is used because the Gaussian may have chopped tails.
        gaussian_area = abs(amp)*sigma*np.sqrt(2*np.pi)*math.erf(n_sigmas)
        area = gaussian_area + abs(amp)*width

        target_area = theta / (np.pi / 2.) * area

        if target_area > gaussian_area:
            width = (target_area - gaussian_area)/abs(amp)
            duration = math.ceil((width+n_sigmas*sigma) / sample_mult) * sample_mult
            return Play(GaussianSquare(amp=amp, width=width, sigma=sigma, duration=duration),
                        channel=instruction.channel)
        else:
            amp_scale = target_area / gaussian_area
            duration = math.ceil(n_sigmas*sigma / sample_mult) * sample_mult
            return Play(GaussianSquare(amp=amp*amp_scale, width=0, sigma=sigma, duration=duration),
                        channel=instruction.channel)

    @staticmethod
    def name(theta: float, phase_delta: float = 0.0, amp_increase: float = 0.) -> str:
        """
        Args:
            theta: Rotation angle of the parametric CZ gate.
            phase_delta: Multiplies the CR90_u pulses samples by np.exp(1.0j*phase_delta).
            amp_increase: Multiplies the CR90_u samples by (1. + amp_increase).
        """
        schedule_name = 'cz_{:.0f}mrad'.format(theta * 1e3)
        if phase_delta != 0.:
            schedule_name += '_{:.0f}mrad'.format(phase_delta * 1e3)
        if amp_increase != 0.:
            schedule_name += '_{:.0f}amp'.format(amp_increase * 1e3)

        return schedule_name

    def parameterized_cx(self, theta: float, q1: int, q2: int, phase_delta: float = 0.,
                         amp_increase: float = 0.):
        """
        Args:
            theta: Rotation angle of the parameterized CZ gate.
            q1: First qubit.
            q2: Second qubit.
            phase_delta: Multiplies the CR90_u pulses samples by np.exp(1.0j*phase_delta).
            amp_increase: Multiplies the amplitude of the samples by (1. + amp_increase).
        """
        cx_sched = self._inst_map.get('cx', qubits=(q1, q2))
        zx_theta = Schedule(name=self.name(theta, phase_delta))

        crs = []
        comp_tones = []
        shift_phases = []
        control = None
        target = None

        for time, inst in cx_sched.instructions:

            if isinstance(inst, ShiftPhase) and time == 0:
                shift_phases.append(ShiftPhase(-theta, inst.channel))

            # Identify the CR pulses.
            if isinstance(inst, Play) and not isinstance(inst, ShiftPhase):
                if isinstance(inst.channel, ControlChannel):
                    crs.append((time, inst))

            # Identify the compensation tones.
            if isinstance(inst.channel, DriveChannel) and not isinstance(inst, ShiftPhase):
                if isinstance(inst.pulse, GaussianSquare):
                    comp_tones.append((time, inst))
                    target = inst.channel.index
                    control = q1 if target == q2 else q2

        if control is None:
            raise QiskitError('Control qubit is None.')
        if target is None:
            raise QiskitError('Target qubit is None.')

        # ntb - we do not care about the initial x90p
        # Id the X90 gate at the start of the schedule and rescale it.
        #x90 = cx_sched.filter(time_ranges=[(0, crs[0][0])], channels=[DriveChannel(target)]).instructions
        #if len(x90) != 1:
        #    raise QiskitError('Could not extract initial X90.')
        #x90 = x90[0][1]

        echo_x = self._inst_map.get('x', qubits=control)

        # Build the schedule
        for inst in shift_phases:
            zx_theta = zx_theta.insert(0, inst)

        # ntb - these pre-pulses we do not want
        #zx_theta = zx_theta.insert(0, self._inst_map.get('x', qubits=control))
        #zx_theta = zx_theta.insert(0, self.rescale_amp(x90, theta))

        # Stretch/compress the CR gates and compensation tones
        cr1 = self.rescale_cr_inst(crs[0][1], theta)
        cr2 = self.rescale_cr_inst(crs[1][1], theta)
        comp1 = self.rescale_cr_inst(comp_tones[0][1], theta)
        comp2 = self.rescale_cr_inst(comp_tones[1][1], theta)

        if theta != 0.0:
            zx_theta = zx_theta.insert(0, cr1)
            zx_theta = zx_theta.insert(0, comp1)
            zx_theta = zx_theta.insert(0 + comp1.duration, echo_x)
            time = comp1.duration + echo_x.duration
            zx_theta = zx_theta.insert(time, cr2)
            zx_theta = zx_theta.insert(time, comp2)
        else:
            zx_theta = zx_theta.insert(0, echo_x)

        return zx_theta

    @staticmethod
    def _contains(schedule: Schedule, name: str) -> bool:
        """Check the schedule to see if it contains a given pulse by name."""
        for inst in schedule.instructions:
            if inst[1].name is None:
                continue

            if name in inst[1].name:
                return True

        return False

    def build_cz_schedule(self, q1: int, q2: int, theta: float, phase_delta: float = 0.,
                          amp_increase: float = 0.):
        """
        Create the pulse schedule for a CZ gate with custom angle.
        The CZ schedule is build from the CNOT schedule in which there is a YM_d rotation.

        Args:
            q1: control qubit for the CR gate.
            q2: target qubit for the CR gate.
            theta: desired angle of the CZ gate.
            phase_delta: Multiplies the CR90_u pulses samples by np.exp(1.0j*phase_delta).
            amp_increase: Multiplies the CR90_u samples by (1. + amp_increase).
        """

        # The CX schedules between (q1, q2) is not the same as (q2, q1).
        # Get the CX schedule which ends with a GaussianSquare pulse
        cx_sched = self._inst_map.get('cx', qubits=(q1, q2))

        if not isinstance(cx_sched.instructions[-1][1].pulse, GaussianSquare):
            cx_sched = self._inst_map.get('cx', qubits=(q2, q1))
            q1, q2 = q2, q1

        if not isinstance(cx_sched.instructions[-1][1].pulse, GaussianSquare):
            raise QiskitError('Could not parse CX schedule for qubits (%i,%i).' % (q1, q2))

        cz_theta = Schedule(name='cz_%i_mrad' % int(theta * 1e3))
        cz_theta |= self._inst_map.get('u2', P0=0.0, P1=np.pi, qubits=[q2])

        cx_theta = self.parameterized_cx(theta, q1, q2, phase_delta=phase_delta,
                                         amp_increase=amp_increase)
        cz_theta |= cx_theta << cz_theta.duration

        # Hadamard
        cz_theta |= self._inst_map.get('u2', P0=0.0, P1=np.pi, qubits=[q2]) << cz_theta.duration

        return cz_theta

    def cz_gate(self, q1, q2, theta: float, phase_delta: float = 0.,
                amp_increase: float = 0.):
        """
        Creates a CZ and inserts the schedule into the instruction
        schedule map of the backend to which the builder is tied.

        Args:
            q1: first qubit.
            q2: second qubit.
            theta: Rotation angle of the CZ gate.
            phase_delta: Multiplies the CR90_u pulses samples by np.exp(1.0j*phase_delta).
            amp_increase: Multiplies the CR90_u samples by (1. + amp_increase).
        """
        cz_name = self.name(theta, phase_delta=phase_delta, amp_increase=amp_increase)

        if not self._inst_map.has(cz_name, (q1, q2)):
            cz_schedule = self.build_cz_schedule(q1, q2, theta, phase_delta=phase_delta
                                                 , amp_increase=amp_increase)
            self._inst_map.add(cz_name, (q1, q2), cz_schedule)
            self._inst_map.add(cz_name, (q2, q1), cz_schedule)

            if cz_name not in self._config.basis_gates:
                self._config.basis_gates.append(cz_name)

        return Gate(cz_name, 2, [])

    def register_czs(self, circ: QuantumCircuit):
        """
        Parses the given quantum circuit and creates the schedules
        for all the CPhase gates.

        Args:
            circ: A QuantumCircuit for which schedules will be
                created for CPhaseGates.
        """
        for gate, qubits, _ in circ.data:
            if isinstance(gate, CPhaseGate):
                theta = gate.params[0]
                q0 = qubits[0].index
                q1 = qubits[1].index
                schedule = self.build_cz_schedule(q0, q1, theta)
                circ.add_calibration(gate, qubits, schedule)

        # TODO Could add a modulo pi on the CRZ parameter
