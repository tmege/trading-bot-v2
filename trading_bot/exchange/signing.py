import logging
import time

import msgpack
from Crypto.Hash import keccak as crypto_keccak
from eth_account import Account

log = logging.getLogger(__name__)

DOMAIN_TYPE_HASH = None
AGENT_TYPE_HASH = None
DOMAIN_SEPARATOR = None


def _keccak256(data: bytes) -> bytes:
    h = crypto_keccak.new(digest_bits=256)
    h.update(data)
    return h.digest()


def _init_constants() -> None:
    global DOMAIN_TYPE_HASH, AGENT_TYPE_HASH, DOMAIN_SEPARATOR

    AGENT_TYPE_HASH = _keccak256(b"Agent(string source,bytes32 connectionId)")
    DOMAIN_TYPE_HASH = _keccak256(
        b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
    )
    DOMAIN_SEPARATOR = _keccak256(
        DOMAIN_TYPE_HASH
        + _keccak256(b"Exchange")
        + _keccak256(b"1")
        + int.to_bytes(1337, 32, "big")
        + b"\x00" * 12 + b"\x00" * 20
    )


_init_constants()


class Signer:
    def __init__(self, private_key: str, is_testnet: bool = False):
        self._account = Account.from_key(private_key)
        self.address = self._account.address
        self._source = "b" if is_testnet else "a"
        log.info(f"Signer initialized for {self.address[:10]}...")

    def sign(
        self, action: dict, vault_address: str | None = None, nonce: int | None = None
    ) -> dict:
        if nonce is None:
            nonce = int(time.time() * 1000)

        connection_id = self._build_connection_id(action, nonce, vault_address)
        digest = self._build_digest(connection_id)

        signed = self._account.unsafe_sign_hash(digest)

        return {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v,
        }

    def _build_connection_id(
        self, action: dict, nonce: int, vault_address: str | None
    ) -> bytes:
        packed = msgpack.packb(action)
        data = packed + nonce.to_bytes(8, "big")

        if vault_address:
            data += b"\x01" + bytes.fromhex(vault_address[2:] if vault_address.startswith("0x") else vault_address)
        else:
            data += b"\x00"

        return _keccak256(data)

    def _build_digest(self, connection_id: bytes) -> bytes:
        source_hash = _keccak256(self._source.encode())

        struct_hash = _keccak256(
            AGENT_TYPE_HASH + source_hash + connection_id
        )

        return _keccak256(
            b"\x19\x01" + DOMAIN_SEPARATOR + struct_hash
        )

    def get_nonce(self) -> int:
        return int(time.time() * 1000)
