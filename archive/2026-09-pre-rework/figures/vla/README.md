# VLA figures

Empty by design. `figures/initial/` holds everything from the image-dataset phase
(ColorMNIST, CelebA, Shapes3D, the latent DiT, and the multi-object formulation); this
directory is where the embodied-policy figures go, so the two phases never share a
namespace and a reader can tell at a glance which claim a figure belongs to.

`docs/VLA_PATH.md` is the staged plan those figures will document, in order:

  Stage A   does the policy's conditioning table FACTORISE?  (gates everything else)
  Stage B   port `diag_backbone_conventions` before any result is believed
  Stage C   hold out a task COMBINATION, never a primitive
  Stage C-bis  a scene of k known objects as a composition over addresses (§7bb)
  Stage D   give the conditioning pathway its own learning rate from the first run (§5t-ter)
  Stage E   judge on rollout success, never on the training loss
  Stage F   calibrate the router's operating point per backbone (§6c-quinquies)
