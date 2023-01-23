from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
import numpy as np
import pandas as pd
import torch
from .secagg import public_key_to_bytes, bytes_to_public_key, generate_key_pairs, generate_shared_key, \
    quantize, reverse_quantize, encrypt, decrypt, private_key_to_bytes, bytes_to_private_key
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
        self.cache_pth = client_dir / 'cache.pkl'
        self.shared_secret_dict = {}
        self.shared_seed_dict = {}
        self.secret_key_dict = {}
        self.rnd_cnt = 0
        self.stage = -1
        # self.weights = np.random.rand(28 if cid == 'swift' else 26)
        self.partial_grad = np.zeros(26, dtype=int)
        self.beta = np.array(0)

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

    def setup(self, server_round, parameters, config):
        rnd = server_round
        # rnd 1
        # generate keys and reply with public keys
        if rnd == 1:
            logger.info(f'client {self.cid}: generating key pairs')
            for cid in config:
                if cid == self.cid:
                    continue
                sk, pk = generate_key_pairs()
                self.secret_key_dict[cid] = private_key_to_bytes(sk)
                config[cid] = public_key_to_bytes(pk)
            config.pop(self.cid)
            t = ([], 0, config)
            logger.info(f'client {self.cid}: replying {str(config.keys())}')
            self.setup_round1(parameters, config, t)
            return t
        # rnd 2
        # generate shared secrets
        if rnd == 2:
            logger.info(f'client {self.cid}: receiving public keys from {str(config.keys())}')
            # logger.info(f'client {self.cid}: keys of secret key dict {str(self.secret_key_dict.keys())}')
            for cid, pk_bytes in config.items():
                sk = bytes_to_private_key(self.secret_key_dict[cid])
                pk = bytes_to_public_key(pk_bytes)
                shared_key = generate_shared_key(sk, pk)
                self.shared_secret_dict[cid] = shared_key
                seed32 = 0
                for i in range(0, len(shared_key), 4):
                    seed32 ^= int.from_bytes(shared_key[i:i + 4], 'little')
                self.shared_seed_dict[cid] = np.array(seed32, dtype=np.int32)

            logger.info(f'client {self.cid}: shared seed {str(self.shared_seed_dict)}')
            t = ([], 0, {})
            self.setup_round2(parameters, config, t)
            self.secret_key_dict = {}
            return t

    def setup_round1(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        pass

    def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        pass

    def stage0(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        pass

    def stage1(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        pass

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        pass

    def get_vars(self):
        return vars(self)

    def cache(self):
        # for k, v in vars(self).items():
        #     logger.info(f"client {self.cid}: saving {k}")
        #     torch.save(v, self.cache_pth)
        with open(self.cache_pth, 'wb') as f:
            pickle.dump(self.get_vars(), f)
        # torch.save(vars(self), self.cache_pth)

    def reload(self):
        if self.cache_pth.exists():
            logger.info(f'client {self.cid}: reloading from {str(self.cache_pth)}')
            with open(self.cache_pth, 'rb') as f:
                self.__dict__.update(pickle.load(f))
            # self.__dict__.update(torch.load(self.cache_pth))

    def fit(
            self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        rnd = config.pop('round')
        if rnd > 1:
            self.reload()
        if rnd > self.rnd_cnt:
            self.rnd_cnt = rnd
        else:
            logger.error(f"Round error at Client {self.cid}: round {rnd} should have been completed.")
            raise RuntimeError(f"Round error: round {rnd} should have been completed.")

        if rnd <= 2:
            ret = self.setup(rnd, parameters, config)
            self.cache()
            return ret

        # rnd 3 -> N, joint training
        self.check_stage(config.pop('stage'))
        # stage 0
        # SWIFT ONLY, reply with batch selection, w1 * proba, weights
        if self.stage == 0:
            ret = self.stage0(rnd, parameters, config)
            if 'stop' in config:
                logger.info('swift client: stop signal is detected. abort federated training')
        # stage 1
        # BANK ONLY, reply with masked wx
        elif self.stage == 1:
            ret = self.stage1(rnd, parameters, config)
        # stage 2
        elif self.stage == 2:
            ret = self.stage2(rnd, parameters, config)
        else:
            raise AssertionError()

        self.cache()
        return ret


