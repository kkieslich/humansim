"""Diagnostics for the crowd model, test-only: not used by the node, not for study runs.

`probe.ForceProbe` records, for chosen walker types, the force terms, every neighbour's push and the velocity at each stage
of a tick (planner, kinematic limits, locomotion, contact separation). `oracles` holds counterfactual behaviours used to
test hypotheses about stalls (ideal yielding of adults to wheelchair users). Both attach to a running `AgentManager` from
a driver script by wrapping its methods; with nothing attached the simulator is unchanged.
"""
