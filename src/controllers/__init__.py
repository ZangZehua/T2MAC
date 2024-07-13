REGISTRY = {}

from .basic_controller import BasicMAC
from .vffac_controller import VffacMAC
from .tmac_vffac_controller import VffacMAC as TMACVffacMAC
from .tmac_full_comm_controller import VffacMAC as TFCMAC
from .tmac_p2p_comm_controller import VffacMAC as P2PMAC
from .tmac_comm_rate_controller import VffacMAC as CRMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['vffac_mac'] = VffacMAC
REGISTRY['tmac_vffac_mac'] = TMACVffacMAC
REGISTRY['tmac_full_comm_mac'] = TFCMAC
REGISTRY['tmac_p2p_comm_mac'] = P2PMAC
REGISTRY['tmac_comm_rate_mac'] = CRMAC