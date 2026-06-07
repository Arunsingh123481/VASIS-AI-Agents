# db/atom_store.py
# Atomic store for fast in-memory atom lookups, page queries, and neighborhood expansion

class AtomStore:
    def __init__(self, atoms: list[dict]):
        # Map atom_id -> atom dict
        # Make sure atom_id is stored as int
        self.atoms = {int(a["atom_id"]): a for a in atoms}

    def get(self, atom_id: int) -> dict | None:
        """Retrieve a single atomic block by ID."""
        return self.atoms.get(int(atom_id))

    def get_many(self, ids: list[int]) -> list[dict]:
        """Retrieve multiple atomic blocks by their IDs."""
        return [self.atoms[int(i)] for i in ids if int(i) in self.atoms]

    def get_in_page_range(self, start: int, end: int) -> list[dict]:
        """Retrieve all atomic blocks within a specific page number range."""
        return [
            a for a in self.atoms.values()
            if start <= a.get("page_number", a.get("page_num", 0)) <= end
        ]

    def get_neighbours(self, atom_id: int, radius: int) -> list[dict]:
        """Retrieve neighboring atomic blocks within a specified radius."""
        atom_id = int(atom_id)
        return [
            self.atoms[i]
            for i in range(atom_id - radius, atom_id + radius + 1)
            if i != atom_id and i in self.atoms
        ]

    def all(self) -> list[dict]:
        """Retrieve all stored atomic blocks."""
        return list(self.atoms.values())

    def get_last_n_pages(self, n: int) -> list[dict]:
        """Retrieve all atoms from the last N pages of the document — used for bibliography targeting."""
        all_pages = sorted(set(
            a.get("page_number", a.get("page_num", 0))
            for a in self.atoms.values()
        ))
        if not all_pages:
            return []
        tail_pages = set(all_pages[-n:])
        return [
            a for a in self.atoms.values()
            if a.get("page_number", a.get("page_num", 0)) in tail_pages
        ]
