from __future__ import annotations

import hashlib
import hmac
import time
import unittest
from unittest.mock import patch

from src import request_auth


class SignedRequestTests(unittest.TestCase):
    def setUp(self):
        self.secret = "test-secret-that-is-at-least-32-characters-long"
        self.body = b'{"question":"hello"}'
        self.timestamp = str(int(time.time()))
        self.nonce = "123e4567-e89b-42d3-a456-426614174000"
        digest = hashlib.sha256(self.body).hexdigest()
        canonical = f"POST\n/chat\n{self.timestamp}\n{self.nonce}\n{digest}"
        self.signature = hmac.new(
            self.secret.encode(), canonical.encode(), hashlib.sha256
        ).hexdigest()
        self.secret_patch = patch.object(
            request_auth, "DAMI_RAG_SHARED_SECRET", self.secret
        )
        self.secret_patch.start()
        self.addCleanup(self.secret_patch.stop)

    @patch("src.request_auth._claim_nonce", return_value=True)
    def test_valid_signature_is_accepted(self, claim):
        request_auth.verify_signed_request(
            method="POST", path="/chat", raw_body=self.body,
            timestamp=self.timestamp, nonce=self.nonce, signature=self.signature,
        )
        claim.assert_called_once_with(self.nonce, int(self.timestamp))

    @patch("src.request_auth._claim_nonce", return_value=True)
    def test_tampered_body_is_rejected(self, claim):
        with self.assertRaises(request_auth.RequestAuthError):
            request_auth.verify_signed_request(
                method="POST", path="/chat", raw_body=b'{"question":"changed"}',
                timestamp=self.timestamp, nonce=self.nonce, signature=self.signature,
            )
        claim.assert_not_called()

    @patch("src.request_auth._claim_nonce", return_value=True)
    def test_expired_timestamp_is_rejected(self, claim):
        with self.assertRaises(request_auth.RequestAuthError):
            request_auth.verify_signed_request(
                method="POST", path="/chat", raw_body=self.body,
                timestamp=str(int(time.time()) - 1000), nonce=self.nonce,
                signature=self.signature,
            )
        claim.assert_not_called()

    @patch("src.request_auth._claim_nonce", return_value=False)
    def test_replayed_nonce_is_rejected(self, claim):
        with self.assertRaises(request_auth.RequestAuthError):
            request_auth.verify_signed_request(
                method="POST", path="/chat", raw_body=self.body,
                timestamp=self.timestamp, nonce=self.nonce, signature=self.signature,
            )

    @patch("src.request_auth._claim_nonce", return_value=True)
    def test_wrong_path_is_rejected(self, claim):
        with self.assertRaises(request_auth.RequestAuthError):
            request_auth.verify_signed_request(
                method="POST", path="/other", raw_body=self.body,
                timestamp=self.timestamp, nonce=self.nonce, signature=self.signature,
            )
        claim.assert_not_called()


if __name__ == "__main__":
    unittest.main()
