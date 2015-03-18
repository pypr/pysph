This directory contains a bunch of examples that demonstrate rigid body
motion and interaction both with other rigid bodies and rigid-fluid
coupling.

The demos here are only proofs of concept.  They need work to make sure
that the physics is correct, the equations correct and produce the right
numbers. In particular,

 - the rigid_block in tank does not work without the pressure rigid body
   equation which is incorrect.

 - the formulation and parameters used for the rigid body collision is not
   tested if it conserves energy and works correctly in all cases.  The choice
   of parameters is currently ad-hoc.

 - the rigid-fluid coupling should also be looked at a bit more carefully with
   proper comparisons to well-known results.

Right now, it looks pretty and is a reasonable start.
