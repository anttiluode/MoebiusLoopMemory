# Perception Lab × MoebiusLoopMemory: Concerto

The [original ECG graph](https://github.com/anttiluode/GeometricNeuronOriginReview/blob/main/ecg.json) was a closed loop: checkerboard image -> 16x16 downsample -> four coarse pixel values -> 50-step variance homeostat -> checkerboard square size. The four samples were **not** Fourier eigenmodes.

A native Perception Lab adapter is now in [PerceptionLab/nodes/moebiusloopmemorynode.py](https://github.com/anttiluode/PerceptionLab/blob/main/nodes/moebiusloopmemorynode.py). It imports a copy of this repository's `loopmemory.py` and uses the **same** `LoopMemory.advance()` equation as the webcam experiment.

Two saved graphs:
- [Passive observer](https://github.com/anttiluode/PerceptionLab/blob/main/graph%20workflows/ecg_moebius_observer.json): records recurrent history of the image vector but does not feed the memory back into the ECG controller.
- [Closed-loop feedback](https://github.com/anttiluode/PerceptionLab/blob/main/graph%20workflows/ecg_moebius_feedback.json): returns a deliberately small memory signal to the homeostat's gain input.

[Setup and falsification protocol](https://github.com/anttiluode/PerceptionLab/blob/main/docs/ECG_MOEBIUS_CONCERTO.md).

## The experimental point

The original ECG feedback changes the generated visual pattern. The new memory node **encodes the succession of observed patterns**. Adding a weak feedback edge changes the *generation process* on the next step.

A feedback-induced difference in pulse timing is not itself a useful computation; the real benchmark is whether a linear readout can recognize the temporal order of an external signal, under equal resources and held-out data, better than the original controller, a rolling average and an ungated loop.

**External input versus generated activity must be separated.** If a feedback controller oscillates with no new camera input, this is autonomous controller dynamics—not evidence of external sensory memory.

### Tested versus untested

The core numerical model, camera encoder and an analogous headless Perception Lab node adapter passed local synthetic tests. The saved graphs were checked to preserve original ECG edges and add precisely one observer edge or one extra feedback edge. The PyQt GUI integration and its waveform effects have **not been experimentally verified** on the user's PC.

The 16D loop is a numerical recurrent network, not a physical cortical wave or a topology proof. In the prior tests, ordinary cyclic recurrence matched the signed/Möbius-like recurrence in useful nonlinear memory tasks.
