//! Native external-sampling MCCFR for Leduc Hold'em.
//!
//! This is a port of the hot loop in `mccfr.py` + `leduc.py`, not a
//! reimplementation with its own ideas. It is written to produce **bit-identical**
//! results to the pure-Python path for the same seed, which is what makes it safe
//! to swap in underneath an unchanged `MCCFRTrainer` interface. Three things are
//! required for that, and each is called out where it happens below:
//!
//! 1. **The RNG.** CPython's `random.Random` is Mersenne Twister. Rather than
//!    reproducing its seeding, the Python side hands over `rng.getstate()` and
//!    takes the state back afterwards, so the stream simply continues here. The
//!    draws themselves (`random()`, `_randbelow`, `shuffle`, `choice`) are
//!    reimplemented to match CPython's exact bit manipulation.
//!
//! 2. **The dot product is a plain sequential sum**, never `mul_add`. This
//!    matters: an FMA contracts the multiply and add into one rounding, which
//!    gives a different result from separate IEEE-754 multiply and add. The
//!    Python side avoids `np.dot` for the same reason -- OpenBLAS picks its
//!    kernel from the CPU at runtime, so it fuses on some machines and not
//!    others. Using only exactly-specified multiply and add makes the result
//!    identical on every conforming machine, in both languages.
//!
//! 3. **Operation order.** Regret matching, the cumulative sums used for
//!    opponent sampling, and the regret/strategy updates all accumulate in the
//!    same order as the Python, because float addition is not associative.
//!
//! The game tree is Leduc exactly as `leduc.py` defines it: 6 cards (two each of
//! J/Q/K), two betting rounds, bet sizes 2 then 4, a cap of 2 raises per round.

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Mersenne Twister, matching CPython's _randommodule.c
// ---------------------------------------------------------------------------

const MT_N: usize = 624;
const MT_M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;

struct Mt {
    state: [u32; MT_N],
    index: usize,
}

impl Mt {
    fn new() -> Self {
        Mt { state: [0; MT_N], index: MT_N }
    }

    /// Load a state tuple from Python's `random.Random.getstate()[1]`:
    /// 624 state words followed by the index.
    fn load(&mut self, words: &[u32]) {
        self.state.copy_from_slice(&words[..MT_N]);
        self.index = words[MT_N] as usize;
    }

    fn save(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(MT_N + 1);
        out.extend_from_slice(&self.state);
        out.push(self.index as u32);
        out
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            let mt = &mut self.state;
            for kk in 0..(MT_N - MT_M) {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + MT_M] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            for kk in (MT_N - MT_M)..(MT_N - 1) {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + MT_M - MT_N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            let y = (mt[MT_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// CPython's `random()`: 53 bits from two 32-bit draws.
    #[inline]
    fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// CPython's `getrandbits(k)` for 1 <= k <= 32.
    #[inline]
    fn getrandbits(&mut self, k: u32) -> u32 {
        self.next_u32() >> (32 - k)
    }

    /// CPython's `Random._randbelow_with_getrandbits`.
    #[inline]
    fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let k = 32 - n.leading_zeros(); // exactly Python's n.bit_length()
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Leduc game tree (mirrors leduc.py)
// ---------------------------------------------------------------------------

// Round histories. 0..5 are decision points, 6..10 are round-over (a call
// closed the round). A fold does NOT extend the history in leduc.py; it only
// sets `folded`, so there are no fold histories here either.
const HIST_STRS: [&str; 11] = ["", "c", "r", "cr", "rr", "crr", "cc", "rc", "crc", "rrc", "crrc"];
const N_DECISION: usize = 6;
const FIRST_OVER: u8 = 6;

// Action codes: 0 = check/call, 1 = fold, 2 = bet/raise.
// Legal actions per decision history, in the same order as leduc._LEGAL so
// that regret vector indices line up with the Python side.
const LEGAL: [[u8; 3]; N_DECISION] = [
    [0, 2, 255], // ""     -> ["c", "r"]
    [0, 2, 255], // "c"    -> ["c", "r"]
    [0, 1, 2],   // "r"    -> ["c", "f", "r"]
    [0, 1, 2],   // "cr"   -> ["c", "f", "r"]
    [0, 1, 255], // "rr"   -> ["c", "f"]   (raise cap)
    [0, 1, 255], // "crr"  -> ["c", "f"]   (raise cap)
];
const NACT: [usize; N_DECISION] = [2, 2, 3, 3, 2, 2];
// NEXT[hist][action] for check/call and bet/raise; fold keeps the history.
const NEXT: [[u8; 3]; N_DECISION] = [
    [1, 255, 2],
    [6, 255, 3],
    [7, 255, 4],
    [8, 255, 5],
    [9, 255, 255],
    [10, 255, 255],
];
const FACING_BET: [bool; N_DECISION] = [false, false, true, true, true, true];
const ACTOR: [u8; N_DECISION] = [0, 1, 1, 0, 0, 1]; // len(history) % 2
const BET_SIZE: [i32; 2] = [2, 4];
const ANTE: i32 = 1;

const N_INFO: usize = 288; // 18 preflop + 270 flop

#[derive(Clone, Copy)]
struct St {
    cards: [u8; 2],
    deck4: [u8; 4], // remaining deck in shuffled order; choice() indexes into it
    public: i8,     // -1 until dealt
    round_idx: usize,
    h0: u8,
    h1: u8,
    contrib: [i32; 2],
    folded: i8, // -1 = nobody
}

#[inline]
fn utility(st: &St, player: usize) -> f64 {
    let opp = 1 - player;
    let pot = st.contrib[0] + st.contrib[1];
    if st.folded >= 0 {
        return if st.folded as usize == player {
            -st.contrib[player] as f64
        } else {
            st.contrib[opp] as f64
        };
    }
    let my_card = st.cards[player] as i8;
    let opp_card = st.cards[opp] as i8;
    let my_pair = my_card == st.public;
    let opp_pair = opp_card == st.public;
    let win = if my_pair && !opp_pair {
        true
    } else if opp_pair && !my_pair {
        false
    } else if my_card != opp_card {
        my_card > opp_card
    } else {
        return 0.0; // identical ranks, neither pairs the board: split
    };
    if win {
        (pot - st.contrib[player]) as f64
    } else {
        -st.contrib[player] as f64
    }
}

#[inline]
fn next_state(st: &St, action: u8, hist_id: usize) -> St {
    let mut ns = *st;
    let player = ACTOR[hist_id] as usize;
    if action == 1 {
        ns.folded = player as i8;
        return ns;
    }
    let opp = 1 - player;
    if action == 0 {
        if FACING_BET[hist_id] {
            ns.contrib[player] = ns.contrib[opp];
        }
    } else {
        let to_call = (ns.contrib[opp] - ns.contrib[player]).max(0);
        ns.contrib[player] += to_call + BET_SIZE[st.round_idx];
    }
    let nh = NEXT[hist_id][action as usize];
    if st.round_idx == 0 {
        ns.h0 = nh;
    } else {
        ns.h1 = nh;
    }
    ns
}

/// Dense index for an information set. Preflop: card * 6 + history. Flop:
/// 18 + (((card * 3 + board) * 5 + preflop_line) * 6 + history).
#[inline]
fn info_index(st: &St, player: usize, hist_id: usize) -> usize {
    let card = st.cards[player] as usize;
    if st.round_idx == 0 {
        card * N_DECISION + hist_id
    } else {
        let line = st.h0 as usize - FIRST_OVER as usize;
        let board = st.public as usize;
        18 + (((card * 3 + board) * 5 + line) * N_DECISION + hist_id)
    }
}

/// The `leduc.py` information-set key for a dense index, so the Python side
/// never has to duplicate the index scheme.
fn info_key(idx: usize) -> String {
    if idx < 18 {
        let card = idx / N_DECISION;
        let hist = idx % N_DECISION;
        format!("{}|-|{}/", card, HIST_STRS[hist])
    } else {
        let mut rest = idx - 18;
        let hist = rest % N_DECISION;
        rest /= N_DECISION;
        let line = rest % 5;
        rest /= 5;
        let board = rest % 3;
        let card = rest / 3;
        format!(
            "{}|{}|{}/{}",
            card,
            board,
            HIST_STRS[FIRST_OVER as usize + line],
            HIST_STRS[hist]
        )
    }
}

fn info_actions(idx: usize) -> Vec<String> {
    let hist = if idx < 18 { idx % N_DECISION } else { (idx - 18) % N_DECISION };
    (0..NACT[hist])
        .map(|i| match LEGAL[hist][i] {
            0 => "c".to_string(),
            1 => "f".to_string(),
            _ => "r".to_string(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Node {
    seen: bool,
    n: usize,
    regret: [f64; 3],
    strategy_sum: [f64; 3],
    synced: u64,
}

impl Default for Node {
    fn default() -> Self {
        Node { seen: false, n: 0, regret: [0.0; 3], strategy_sum: [0.0; 3], synced: 0 }
    }
}

/// Mirrors `mccfr._regret_matching`, including the accumulation order.
#[inline]
fn regret_matching(regret: &[f64; 3], n: usize) -> [f64; 3] {
    let mut pos = [0.0f64; 3];
    for i in 0..n {
        pos[i] = if regret[i] >= 0.0 { regret[i] } else { 0.0 };
    }
    let mut total = 0.0f64;
    for i in 0..n {
        total += pos[i];
    }
    let mut out = [0.0f64; 3];
    if total > 0.0 {
        for i in 0..n {
            out[i] = pos[i] / total;
        }
    } else {
        let u = 1.0 / n as f64;
        for i in 0..n {
            out[i] = u;
        }
    }
    out
}

#[pyclass]
struct LeducTrainer {
    rng: Mt,
    nodes: Vec<Node>,
    variant: u8, // 0 plain, 1 plus, 2 dcfr
    alpha: f64,
    beta: f64,
    gamma: f64,
    cum_log_pos: Vec<f64>,
    cum_log_neg: Vec<f64>,
    iteration: u64,
}

impl LeducTrainer {
    fn push_discount(&mut self, t: u64) {
        let tf = t as f64;
        let ta = tf.powf(self.alpha);
        let tb = tf.powf(self.beta);
        let p = self.cum_log_pos[self.cum_log_pos.len() - 1] + (ta / (ta + 1.0)).ln();
        let n = self.cum_log_neg[self.cum_log_neg.len() - 1] + (tb / (tb + 1.0)).ln();
        self.cum_log_pos.push(p);
        self.cum_log_neg.push(n);
    }

    /// Mirrors `MCCFRTrainer._discount_node`: apply every end-of-iteration DCFR
    /// discount that fell due since this node was last visited, in O(1).
    fn discount_node(&mut self, idx: usize, t: u64) {
        let k = self.nodes[idx].synced;
        if k >= t - 1 {
            return;
        }
        let pos_f = (self.cum_log_pos[(t - 1) as usize] - self.cum_log_pos[k as usize]).exp();
        let neg_f = (self.cum_log_neg[(t - 1) as usize] - self.cum_log_neg[k as usize]).exp();
        let sf = ((k + 1) as f64 / t as f64).powf(self.gamma);
        let node = &mut self.nodes[idx];
        for i in 0..node.n {
            let r = node.regret[i];
            if r > 0.0 {
                node.regret[i] = r * pos_f;
            } else if r < 0.0 {
                node.regret[i] = r * neg_f;
            }
            node.strategy_sum[i] *= sf;
        }
        node.synced = t - 1;
    }

    /// `leduc.LeducState.sample_root`: shuffle the deck, deal two cards.
    fn sample_root(&mut self) -> St {
        let mut deck = [0u8, 0, 1, 1, 2, 2];
        // CPython's random.shuffle
        for i in (1..6).rev() {
            let j = self.rng.randbelow((i + 1) as u32) as usize;
            deck.swap(i, j);
        }
        St {
            cards: [deck[0], deck[1]],
            deck4: [deck[2], deck[3], deck[4], deck[5]],
            public: -1,
            round_idx: 0,
            h0: 0,
            h1: 0,
            contrib: [ANTE, ANTE],
            folded: -1,
        }
    }

    fn traverse(&mut self, st: &St, traversing: usize, t: u64) -> f64 {
        // Terminal: somebody folded, or the flop betting round closed.
        if st.folded >= 0 || (st.round_idx == 1 && st.h1 >= FIRST_OVER) {
            return utility(st, traversing);
        }
        // Chance: the preflop round closed, deal the public card. This is
        // `rng.choice(deck_remaining)` on the 4 undealt cards.
        if st.round_idx == 0 && st.h0 >= FIRST_OVER {
            let pick = self.rng.randbelow(4) as usize;
            let mut ns = *st;
            ns.public = st.deck4[pick] as i8;
            ns.round_idx = 1;
            return self.traverse(&ns, traversing, t);
        }

        let hist_id = if st.round_idx == 0 { st.h0 } else { st.h1 } as usize;
        let player = ACTOR[hist_id] as usize;
        let idx = info_index(st, player, hist_id);
        let n = NACT[hist_id];
        {
            let node = &mut self.nodes[idx];
            node.seen = true;
            node.n = n;
        }
        let strategy = regret_matching(&self.nodes[idx].regret, n);

        if player == traversing {
            let mut utils = [0.0f64; 3];
            for i in 0..n {
                let ns = next_state(st, LEGAL[hist_id][i], hist_id);
                utils[i] = self.traverse(&ns, traversing, t);
            }
            // Separate multiply and add, matching the Python exactly. Do NOT
            // use mul_add here: fusing changes the rounding.
            let mut node_util = 0.0f64;
            for i in 0..n {
                node_util += strategy[i] * utils[i];
            }

            match self.variant {
                1 => {
                    let node = &mut self.nodes[idx];
                    for i in 0..n {
                        let r = node.regret[i] + (utils[i] - node_util);
                        node.regret[i] = if r >= 0.0 { r } else { 0.0 };
                        node.strategy_sum[i] += t as f64 * strategy[i];
                    }
                }
                2 => {
                    self.discount_node(idx, t);
                    let node = &mut self.nodes[idx];
                    for i in 0..n {
                        node.regret[i] += utils[i] - node_util;
                        node.strategy_sum[i] += strategy[i];
                    }
                }
                _ => {
                    let node = &mut self.nodes[idx];
                    for i in 0..n {
                        node.regret[i] += utils[i] - node_util;
                        node.strategy_sum[i] += strategy[i];
                    }
                }
            }
            node_util
        } else {
            // Sample one opponent action, reproducing the Python bisect exactly.
            let mut acc = strategy[0];
            for i in 1..n {
                acc += strategy[i];
            }
            let r = self.rng.random() * (acc + 0.0);
            let hi = n - 1;
            let mut a_idx = 0usize;
            let mut acc2 = strategy[0];
            while a_idx < hi && acc2 <= r {
                a_idx += 1;
                acc2 += strategy[a_idx];
            }
            let ns = next_state(st, LEGAL[hist_id][a_idx], hist_id);
            self.traverse(&ns, traversing, t)
        }
    }
}

#[pymethods]
impl LeducTrainer {
    #[new]
    fn new(variant: u8, alpha: f64, beta: f64, gamma: f64) -> Self {
        LeducTrainer {
            rng: Mt::new(),
            nodes: vec![Node::default(); N_INFO],
            variant,
            alpha,
            beta,
            gamma,
            cum_log_pos: vec![0.0],
            cum_log_neg: vec![0.0],
            iteration: 0,
        }
    }

    fn set_rng_state(&mut self, words: Vec<u32>) -> PyResult<()> {
        if words.len() != MT_N + 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected 625 words from random.Random.getstate()[1]",
            ));
        }
        self.rng.load(&words);
        Ok(())
    }

    fn rng_state(&self) -> Vec<u32> {
        self.rng.save()
    }

    #[getter]
    fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Run `iterations` external-sampling MCCFR iterations.
    fn train(&mut self, iterations: u64) {
        for _ in 0..iterations {
            self.iteration += 1;
            let t = self.iteration;
            if self.variant == 2 {
                self.push_discount(t);
            }
            let root = self.sample_root();
            for player in 0..2usize {
                self.traverse(&root, player, t);
            }
        }
    }

    /// (key, actions, regret_sum, strategy_sum, synced) for every visited node.
    fn export(&self) -> Vec<(String, Vec<String>, Vec<f64>, Vec<f64>, u64)> {
        let mut out = Vec::new();
        for idx in 0..N_INFO {
            let node = &self.nodes[idx];
            if !node.seen {
                continue;
            }
            out.push((
                info_key(idx),
                info_actions(idx),
                node.regret[..node.n].to_vec(),
                node.strategy_sum[..node.n].to_vec(),
                node.synced,
            ));
        }
        out
    }
}

#[pymodule]
fn mccfr_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LeducTrainer>()?;
    m.add("__doc__", "Native external-sampling MCCFR inner loop for Leduc Hold'em")?;
    Ok(())
}
