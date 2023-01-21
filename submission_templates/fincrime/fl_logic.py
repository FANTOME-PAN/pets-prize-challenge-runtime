from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
import numpy as np
import pandas as pd
# from torch import nn
from .secagg import public_key_to_bytes, bytes_to_public_key, generate_key_pairs, generate_shared_key, \
    quantize, reverse_quantize, encrypt, decrypt
import pickle

LOGIC_TEST = True
DEBUG = True


def sum_func(x):
    x += 1
    return sum(x)


class TrainClientTemplate(fl.client.NumPyClient):
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
        self.shared_seed_dict = {}
        self.secret_key_dict = {}
        self.rnd_cnt = 0
        self.stage = -1
        # self.weights = np.random.rand(28 if cid == 'swift' else 26)
        # self.partial_grad = np.zeros(26, dtype=int)
        self.beta = np.array(0)
        # if cid == 'swift':
        #     # UNIMPLEMENTED CODE HERE
        #     self.loader = DataLoader(df, {})
        #     self.net = None
        #     self.bank2cid = {}
        #     self.lr = 0.1
        #     self.agg_grad = np.zeros(26)
        #     self.proba = np.array(0)
        # else:
        #     self.cached_flags = np.array(0)

    def check_stage(self, stage):
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

    def setup(self, config):
        rnd = config["round"]
        # rnd 1
        # generate keys and reply with public keys
        if rnd == 1:
            config.pop("round")
            for cid in config:
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
            for cid, pk_bytes in config.items():
                pk = bytes_to_public_key(pk_bytes)
                shared_key = generate_shared_key(
                    sk=self.secret_key_dict[cid],
                    pk=pk
                )
                self.shared_secret_dict[cid] = shared_key
                seed32 = 0
                for i in range(0, len(shared_key), 4):
                    seed32 ^= int.from_bytes(shared_key[i:i + 4], 'little')
                self.shared_seed_dict[cid] = np.array(seed32, dtype=np.int32)
            self.pretrain()
            return [], 0, {}

    def pretrain(self):
        pass

    def stage0(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        pass

    def stage1(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        pass

    def stage2(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        pass

    def fit(
            self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        rnd = config["round"]
        if rnd > self.rnd_cnt:
            self.rnd_cnt = rnd
        else:
            logger.error(f"Round error at Client {self.cid}: round {rnd} should have been completed.")
            raise RuntimeError(f"Round error: round {rnd} should have been completed.")

        if rnd <= 2:
            return self.setup(config)

        # rnd 3 -> N, joint training
        self.check_stage(config['stage'])
        # stage 0
        # SWIFT ONLY, reply with batch selection, w1 * proba, weights
        if self.stage == 0:
            # stage 1
            return self.stage0(parameters, config)
        # BANK ONLY, reply with masked wx
        if self.stage == 1:
            return self.stage1(parameters, config)
        # stage 2
        if self.stage == 2:
            return self.stage2(parameters, config)


