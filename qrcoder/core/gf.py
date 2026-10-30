"""GF(256) arithmetic and a Reed-Solomon decoder for QR error correction.

QR Code uses GF(2^8) with the primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
(0x11D) and generator element alpha = 2.

The decoder follows the classical syndrome -> Berlekamp-Massey -> Chien search
-> Forney pipeline.  It reports *how many* symbols it had to correct, which the
analyser uses as a confidence signal: a block corrected with zero errors is far
more trustworthy than one that needed a full complement of corrections.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

__all__ = ["GF", "ReedSolomonError", "rs_decode", "rs_encode"]

PRIMITIVE = 0x11D
FIELD_SIZE = 256


class GF:
    """Log/antilog tables for GF(256) with polynomial 0x11D."""

    EXP: List[int] = [0] * (FIELD_SIZE * 2)
    LOG: List[int] = [0] * FIELD_SIZE

    @classmethod
    def _build(cls) -> None:
        x = 1
        for i in range(FIELD_SIZE - 1):
            cls.EXP[i] = x
            cls.LOG[x] = i
            x <<= 1
            if x & 0x100:
                x ^= PRIMITIVE
        for i in range(FIELD_SIZE - 1, FIELD_SIZE * 2):
            cls.EXP[i] = cls.EXP[i - (FIELD_SIZE - 1)]

    @classmethod
    def mul(cls, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return cls.EXP[cls.LOG[a] + cls.LOG[b]]

    @classmethod
    def div(cls, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(256)")
        if a == 0:
            return 0
        return cls.EXP[(cls.LOG[a] - cls.LOG[b]) % (FIELD_SIZE - 1)]

    @classmethod
    def pow(cls, a: int, power: int) -> int:
        if power == 0:
            return 1
        if a == 0:
            return 0
        return cls.EXP[(cls.LOG[a] * power) % (FIELD_SIZE - 1)]

    @classmethod
    def inv(cls, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("0 has no inverse in GF(256)")
        return cls.EXP[(FIELD_SIZE - 1) - cls.LOG[a]]


GF._build()


class ReedSolomonError(Exception):
    """Raised when a block cannot be corrected."""


# ---------------------------------------------------------------------------
# Reed-Solomon decoding.
#
# Polynomials in the decoder are stored *lowest degree first* (index == power),
# which keeps Berlekamp-Massey and the Forney evaluation readable.  The public
# API still speaks in codeword order (highest-order symbol first), matching how
# QR blocks are transmitted.
# ---------------------------------------------------------------------------


def _trim(poly: Sequence[int]) -> List[int]:
    out = list(poly)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _poly_mul_lf(a: Sequence[int], b: Sequence[int]) -> List[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        if ca == 0:
            continue
        for j, cb in enumerate(b):
            out[i + j] ^= GF.mul(ca, cb)
    return _trim(out)


def _poly_eval_lf(poly: Sequence[int], x: int) -> int:
    """Horner evaluation of a lowest-degree-first polynomial."""
    y = 0
    for coeff in reversed(poly):
        y = GF.mul(y, x) ^ coeff
    return y


def _poly_eval(poly: Sequence[int], x: int) -> int:
    """Horner evaluation of a highest-degree-first polynomial."""
    y = poly[0] if poly else 0
    for i in range(1, len(poly)):
        y = GF.mul(y, x) ^ poly[i]
    return y


def _syndromes(block: Sequence[int], nsym: int) -> List[int]:
    """Syndromes S_j = block(alpha^j), highest-degree-first block."""
    return [_poly_eval(block, GF.EXP[j]) for j in range(nsym)]


def _berlekamp_massey(synd: Sequence[int], nsym: int) -> Tuple[List[int], int]:
    """Return ``(sigma, L)`` with sigma lowest-degree-first."""
    sigma = [1]
    prev = [1]
    L = 0
    m = 1
    b = 1
    for k in range(nsym):
        d = 0
        for i in range(0, min(L, len(sigma) - 1) + 1):
            if k - i >= 0:
                d ^= GF.mul(sigma[i], synd[k - i])
        if d == 0:
            m += 1
            continue
        coef = GF.div(d, b)
        new = list(sigma)
        need = len(prev) + m
        if len(new) < need:
            new.extend([0] * (need - len(new)))
        for i, pv in enumerate(prev):
            new[i + m] ^= GF.mul(coef, pv)
        if 2 * L <= k:
            L = k + 1 - L
            prev = list(sigma)
            b = d
            m = 1
        else:
            m += 1
        sigma = new
    return _trim(sigma), L


def rs_decode(block: Sequence[int], nsym: int) -> Tuple[List[int], int]:
    """Correct *block*.

    :param block: codeword block, highest-order symbol first.
    :param nsym: number of error correction symbols in the block.
    :returns: ``(corrected_block, errors_corrected)``.
    :raises ReedSolomonError: when the block is beyond repair.
    """
    if nsym <= 0:
        return list(block), 0
    if len(block) > FIELD_SIZE - 1:
        raise ReedSolomonError("block longer than GF(256) permits")

    n = len(block)
    synd = _syndromes(block, nsym)
    if not any(synd):
        return list(block), 0

    sigma, L = _berlekamp_massey(synd, nsym)
    if 2 * L > nsym or len(sigma) - 1 != L:
        raise ReedSolomonError("too many errors to correct")
    if L == 0:
        raise ReedSolomonError("empty error locator for a non-zero syndrome")

    # Error evaluator omega = (S * sigma) mod x^nsym, lowest-degree-first.
    omega = _poly_mul_lf(synd, sigma)[:nsym]

    # Formal derivative of sigma: coefficient i-1 is i*sigma[i], which is
    # sigma[i] for odd i and 0 for even i in GF(2^m).
    dsigma = [sigma[i] if (i % 2) else 0 for i in range(1, len(sigma))]
    if not any(dsigma):
        raise ReedSolomonError("degenerate error locator derivative")

    out = list(block)
    nerr = 0
    for pos in range(n):
        # Error locator X_i = alpha^(n-1-pos) (highest-order-first convention).
        X = GF.EXP[(n - 1 - pos) % (FIELD_SIZE - 1)]
        x_inv = GF.inv(X)
        if _poly_eval_lf(sigma, x_inv) == 0:
            num = _poly_eval_lf(omega, x_inv)
            den = _poly_eval_lf(dsigma, x_inv)
            if den == 0:
                raise ReedSolomonError("zero Forney denominator")
            out[pos] ^= GF.mul(X, GF.div(num, den))
            nerr += 1

    if nerr != L:
        raise ReedSolomonError("error locator found %d roots, expected %d" % (nerr, L))
    if any(_syndromes(out, nsym)):
        raise ReedSolomonError("correction did not clear the syndromes")
    return out, nerr


def _generator_poly(nsym: int) -> List[int]:
    """Generator polynomial for *nsym* EC symbols, highest degree first."""
    g = [1]
    for i in range(nsym):
        nxt = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            nxt[j] ^= c
            nxt[j + 1] ^= GF.mul(c, GF.EXP[i])
        g = nxt
    return g


def rs_encode(data: Sequence[int], nsym: int) -> List[int]:
    """Compute the *nsym* Reed-Solomon symbols for *data* (used by the tests)."""
    gen = _generator_poly(nsym)
    msg = list(data) + [0] * nsym
    for i in range(len(data)):
        coef = msg[i]
        if coef:
            for j, g in enumerate(gen):
                msg[i + j] ^= GF.mul(g, coef)
    return msg[len(data):]
