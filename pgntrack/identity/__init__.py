"""Label-free identity inference on the tracker's output (Sec. 4): interactions, visible-view prototypes, the
exchange-symmetric likelihood ratio kappa D, the EM prior, the MAP rule, and online / batch / cross calibration."""
from .calibration import Calibration, calibrate, decide, em_prior, exchange, fit, kappa_of
from .layer import IdentityLayer, View
from .params import IdentityParams
from .plugin import IdentityPlugin, build_identity_plugin

__all__ = ['Calibration', 'IdentityLayer', 'IdentityParams', 'IdentityPlugin', 'View', 'build_identity_plugin',
           'calibrate', 'decide', 'em_prior', 'exchange', 'fit', 'kappa_of']
