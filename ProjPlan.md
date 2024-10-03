Tentative List:
- Yang, G.R., Joglekar, M.R., Song, H.F. et al. Task representations in neural networks trained to perform many cognitive tasks. Nat Neurosci 22, 297–306 (2019).
- H ̈anni, K., RP, Mendel, J. (n.d.). A starting point for making sense of task structure (in machine learning). LessWrong, (2024).
- Goldfeld, Z., et. al. Estimating Information Flow in Deep Neural Networks, arXiv, (2019). https://arxiv.org/abs/1810.05728
- Lippl, S., et. al. A mathematical theory of relational generalization in transitive inference https://www.biorxiv.org/content/10.1101/2023.08.22.554287v2
- Probe transfer: https://arxiv.org/pdf/2306.03819



### Data
Task Types: 
- Transitive (Relation) Inference; (ones that share the same Relation/Structure)
- Relative Order (The main purpose is to reuse the learnt structure and to analyze some dynamics generated with the structure)

### Model
- Training Strats
  - Interleaving


## Task Types and Requirements (train/test)
### Transitive Inference(TI):
Input: a pair of instances e.g. `(A, B)` or the one-hot-encoded version
Output: 0 or 1 (which one is ‘greater’)
Adj pairs for train data; non-adj pairs for test data; no shared input pairs in both train and test data  (not necessarily have to be exhaustive)

Example: a random pair of `A, B, C, D` 
Total possible pairs $P^2_n$
Adj pair `(A,B)`, `(C,B)` etc.
Non adj pair `(A, C)`, `(A, D)` etc.


### Subset Inclusion(SI):
	Input: a pair of instances e.g. `(A, B)` or the Indicator vector version
	Output: 0 or 1 (which one is a subset of the other)
	Must be the case: either one is a subset of the other(to enforce a parietal-order/linear structure); in train data two sets only differ by one bit in their indicator vectors (similar to Gray Code)


Example: a pair of items in the power set of `{A, B, C, D}`
E.g. the indicator vector of the set `{A,C}` is `[1, 0, 1, 0]`
Total possible pairs $(2^n)P2$ if including the undesired case


### Relative Order(RO):
	Input: a sequence (size = $n$) of pairs of instances, e.g. `[(A, B), (C, A), (A, C)]` or the one-hot-encoded version
	Output: a sequence (size = $n$) of the first k pairs’ ordering, e.g. `[(A, B), (A, A, B, C), (A, A, A, B, C, C)]` or the padded version (so all orderings have the same length)
	If the pair-wise relation should be reused during training? If the input sequence pairs should be exhaustive, to guarantee the correct inferences in the ordering

Example: a sequence of pairs of `A, B, C, D`
For input `[(A, B), (C, A), (A, C)]`, 
the padded version (ordering len = $2n$) of the corresponding output `[(A, B), (A, A, B, C), (A, A, A, B, C, C)]` is 
`[(A, B, x, x, x, x), (A, A, B, C, x, x), (A, A, A, B, C, C)]`

