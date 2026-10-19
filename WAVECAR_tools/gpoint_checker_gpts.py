import numpy as np

class GpointChecker:
    """
    Dual-mode: if `source` is a filename, load two spin-blocks from disk;
    if it's a NumPy array of shape (nk, nG, 3), treat each k-point as a block.
    `check()` returns a list of boolean masks, one per block.
    """

    def __init__(self, source):
        if isinstance(source, str):
            # old file-based path: load spin_blocks from Gpoints.txt
            self.spin_blocks = self._load_spin_blocks(source)
        else:
            arr = np.asarray(source)
            if arr.ndim == 2:
                # single block of shape (nG,3)
                self.spin_blocks = [arr]
            elif arr.ndim == 3:
                # one block per k-point
                # arr.shape = (nk, nG, 3)
                self.spin_blocks = [arr[k] for k in range(arr.shape[0])]
            else:
                raise ValueError(
                    f"GPointChecker: expected array of dim 2 or 3, got {arr.shape}"
                )

        self.removed_indices  = {}
        self.removed_mapping  = {}
        self.filtered_blocks  = []

    def _load_spin_blocks(self, fname):
        blocks, curr = [], []
        with open(fname) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith('#'):
                    if curr:
                        blocks.append(np.array(curr, int))
                        curr = []
                    continue
                curr.append(list(map(int, s.split())))
        if curr:
            blocks.append(np.array(curr, int))
        if len(blocks) != 2:
            raise ValueError(f"Expected 2 spin blocks, got {len(blocks)}")
        return blocks

    def remove_facet_duplicates(self):
        """
        For each block G (shape (nG,3)):
          - find all “−max” duplicates on each axis and drop them
        Returns:
          masks : list of bool arrays, one per block
        """
        self.removed_indices.clear()
        self.removed_mapping.clear()
        self.filtered_blocks.clear()

        masks = []
        for spin_idx, G in enumerate(self.spin_blocks):
            max_abs = np.max(np.abs(G), axis=0)
            to_remove, mapping = set(), {}

            for a in (0,1,2):
                plus_idx  = np.where(G[:,a] ==  max_abs[a])[0]
                minus_idx = np.where(G[:,a] == -max_abs[a])[0]
                others    = [ax for ax in (0,1,2) if ax != a]

                # group minus‐facet by the other two coords
                minus_dict = {}
                for j in minus_idx:
                    key = tuple(G[j, others])
                    minus_dict.setdefault(key, []).append(j)

                # any plus‐facet sharing the same key knocks out those minus‐points
                for i in plus_idx:
                    key = tuple(G[i, others])
                    for j in minus_dict.get(key, []):
                        to_remove.add(j)
                        mapping[j] = i

            removed = np.array(sorted(to_remove), dtype=int)
            keep    = np.setdiff1d(np.arange(len(G)), removed)

            self.removed_indices[spin_idx] = removed
            self.removed_mapping[spin_idx] = mapping
            self.filtered_blocks.append(G[keep])

            # build boolean mask
            mask = np.ones(len(G), dtype=bool)
            mask[removed] = False
            masks.append(mask)

        return masks

    # alias
    check = remove_facet_duplicates


    def summary(self):
        print("Spin‐block shapes:", [b.shape for b in self.spin_blocks])
        removed = self.remove_facet_duplicates()

        for spin_idx, rem in removed.items():
            G = self.spin_blocks[spin_idx]
            if rem.size == 0:
                print(f"Spin {spin_idx}: no facet duplicates")
                continue

            print(f"\nSpin {spin_idx}: removed {len(rem)} facet duplicates")
            for r in rem:
                k = self.removed_mapping[spin_idx][r]
                print(
                    f"  • removed idx={r}, G={G[r].tolist()}    "
                    f"←→ kept idx={k}, G={G[k].tolist()}"
                )


if __name__ == "__main__":
    # FILE mode
    chk = GpointChecker(
        r"dir/run/Gpoints.txt"
    )
    chk.summary()

    # ARRAY mode
    arr = np.random.randint(-3,4,(2,10,3))  # nk=2, nG=10
    chk2 = GpointChecker(arr)
    print(chk2.check(), chk2.filtered_blocks)