from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
import numpy as np
import pandas as pd
from torch import nn
from .secagg import public_key_to_bytes, bytes_to_public_key, generate_key_pairs, generate_shared_key, \
    quantize, reverse_quantize

DEBUG = True


"""=== Utility Functions ==="""


def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return ndarrays_to_parameters([])


# train and return the XGBoost on swift_train
def pretrain_swift_model(swift_df: pd.DataFrame) -> nn.Module:
    raise NotImplementedError()


# return predicted probabilities
def predict_proba(net, x) -> np.ndarray:
    raise NotImplementedError()


def masking(seed_offset, x: np.ndarray, cid, shared_secret_dict: Dict, target_range=(1 << 32)) \
        -> np.ndarray:
    for other_cid, seed in shared_secret_dict:
        np.random.seed(seed + seed_offset)
        msk = np.random.randint(0, target_range, x.shape)
        if cid < other_cid:
            x += msk
        else:
            x -= msk
    return x


class DataLoader:
    def __init__(self, df: pd.DataFrame, bank2cid: Dict[str, str]):
        self.df = df
        self.bank2cid = bank2cid
        pass

    # return input batch to XGBoost and
    # bathed (sender_client_cid, receiver_client_cid, ordering_account, beneficiary_account)
    def next_batch(self) -> List[np.ndarray]:
        raise NotImplementedError()








# class TrainSwiftClient(fl.client.NumPyClient):
#     """Custom Flower NumPyClient class for training."""
#     def __init__(
#         self, cid: str, swift_df: pd.DataFrame, model: SwiftModel, client_dir: Path
#     ):
#         super().__init__()
#         self.cid = cid
#         self.swift_df = swift_df
#         self.model = model
#         self.client_dir = client_dir


"""=== Client Class ==="""


class TrainClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for training."""
    def __init__(
        self, cid: str, df: pd.DataFrame, model, client_dir: Path
    ):
        super().__init__()
        self.cid = cid
        self.df = df
        self.model = model
        self.client_dir = client_dir
        self.shared_secret_dict = {}
        self.secret_key_dict = {}
        self.rnd_cnt = 0
        self.stage = -1
        self.weights = np.random.rand(28)
        if cid == 'swift':
            # UNIMPLEMENTED CODE HERE
            self.loader = DataLoader(df, {})
            self.net = None

    def _check_stage(self, stage):
        # init
        if self.stage == -1:
            if self.cid == 'swift':
                assert stage == 0
            else:
                assert stage == 1
        elif self.stage == 0:
            assert self.cid == 'swift'
            # swift client is not in stage 1
            assert stage == 2
        elif self.stage == 1:
            assert self.cid != 'swift'
            assert stage == 2
        elif self.stage == 2:
            if self.cid == 'swift':
                assert stage == 0
            else:
                # bank client is not in stage 0
                assert stage == 1
        self.stage = stage

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        rnd = config["round"]
        if rnd > self.rnd_cnt:
            self.rnd_cnt = rnd
        else:
            logger.error(f"Round error at Client {self.cid}: round {rnd} should have been completed.")
            raise RuntimeError(f"Round error: round {rnd} should have been completed.")
        # rnd 1
        # generate keys and reply with public keys
        if rnd == 1:
            config.pop("round")
            for cid in config.keys():
                if cid == self.cid:
                    continue
                sk, pk = generate_key_pairs()
                self.secret_key_dict[sk] = sk
                config[cid] = public_key_to_bytes(pk)
            return [], 0, config
        # rnd 2
        # generate shared secrets
        if rnd == 2:
            config.pop("round")
            for cid, pk_bytes in config:
                pk = bytes_to_public_key(pk_bytes)
                seed = generate_shared_key(
                    sk=self.secret_key_dict[cid],
                    pk=pk
                )
                seed32 = 0
                for i in range(0, len(seed), 4):
                    seed32 ^= int.from_bytes(seed[i:i + 4], 'little')
                self.shared_secret_dict[cid] = np.array(seed32, dtype=np.int32)
            if self.cid == 'swift':
                logger.info("swift client: train XGBoost")
                # training code
                # UNIMPLEMENTED CODE HERE
                self.net = pretrain_swift_model(self.df)
            return [], 0, {}
        # rnd 3 -> N, joint training
        self._check_stage(config['stage'])
        # stage 0
        # SWIFT ONLY, reply with batch selection, w1 * proba, weights
        if self.stage == 0:
            x, batch = self.loader.next_batch()
            proba = predict_proba(self.net, x)
            wx = quantize([self.weights[0] * proba], 16, 1 << 24)
            wx = masking(rnd + 1, wx, self.cid, self.shared_secret_dict)
            







def train_client_factory(
    cid: str,
    data_path: Path,
    client_dir: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        data_path (Path): Path to CSV data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    ...
    return TrainClient(...)


"""=== Strategy Class ==="""


class TrainStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for training."""
    def __init__(self, server_dir: Path):
        self.server_dir =server_dir
        self.public_keys_dict = {}
        self.fwd_dict = {}
        self.agg_grad = np.zeros(27)
        self.stage = 0
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return empty_parameters()

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        cid_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict = {"round": server_round}
        logger.info(f"[START] server round {server_round}")
        # rnd 1
        # collect public keys
        if server_round == 1:
            # add all cids to the config_dict as keys
            config_dict = dict(zip(cid_dict.keys(), [0] * len(cid_dict))) | config_dict
            logger.info(f"server's requesting public keys...")
            if DEBUG:
                logger.info(f"send to clients {str(config_dict)}")
            fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            return [(o, fit_ins) for o in cid_dict.items()]
        # rnd 2
        # broadcast public keys, swift train
        if server_round == 2:
            # forward public keys to corresponding clients
            logger.info(f"server's forwarding public keys")
            ins_lst = [(
                proxy,
                FitIns(parameters=empty_parameters(), config=self.fwd_dict[cid] | config_dict)
            ) for cid, proxy in cid_dict]
            self.fwd_dict = {}
            return ins_lst
        # rnd 3 -> N
        # joint train
        config_dict['stage'] = self.stage
        if self.stage == 0:
            logger.info(f"stage 0: sending gradients to swift client")
            fit_ins = FitIns(parameters=ndarrays_to_parameters(self.agg_grad), config=config_dict)
            ins_lst = [(cid_dict['swift'], fit_ins)]
        elif self.stage == 1:
            ins_lst = None
        elif self.stage == 2:
            ins_lst = None
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")

        self.stage = (self.stage + 1) % 3
        return ins_lst


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        if (n_failures := len(failures)) > 0:
            logger.error(f"Had {n_failures} failures in round {server_round}")
            raise Exception(f"Had {n_failures} failures in round {server_round}")
        # rnd 1
        # gather all public keys, and forward them to corresponding client
        if server_round == 1:
            # fwd_dict[to_client][from_client] = public key to to_client generated by from_client
            self.fwd_dict = dict([(o.cid, {}) for o, _ in results])
            for client, res in results:
                for other_cid, pk_bytes in res.metrics:
                    self.fwd_dict[other_cid][client.cid] = pk_bytes
            logger.info(f"[END] server round {server_round}")
            return None, {}
        # rnd 2
        # do nothing
        if server_round == 2:
            logger.info(f"[END] server round {server_round}")
            return None, {}


def train_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    ...
    return TrainStrategy(...)


class TestClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for test."""


def test_client_factory(
    cid: str,
    data_path: Path,
    client_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for test-time
    inference. The federated learning simulation engine will use this function
    to instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages. The
            SWIFT node will always be named 'swift'.
        data_path (Path): Path to CSV test data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.
        preds_format_path (Optional[Path]): Path to CSV file matching the format
            you must write your predictions with, filled with dummy values. This
            will only be non-None for the 'swift' client—bank clients should not
            write any predictions and receive None for this argument.
        preds_dest_path (Optional[Path]): Destination path that you must write
            your test predictions to as a CSV file. This will only be non-None
            for the 'swift' client—bank clients should not write any predictions
            and will receive None for this argument.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    ...
    return TestClient(...)


class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""


def test_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federation rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    ...
    return TestStrategy(...), 1
