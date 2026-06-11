from sortedcontainers import SortedDict
from collections import defaultdict


class Book:
    def __init__(self, strict=False):
        self.orders = {}
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.strict = strict                  
        self.stats = defaultdict(int)         

    def _book_for(self, side):
        return self.bids if side == "B" else self.asks

    def _add_to_level(self, side, price, size):
        book = self._book_for(side)
        book[price] = book.get(price, 0) + size

    def _remove_from_level(self, side, price, size):
        book = self._book_for(side)
        if price not in book:
            self.stats["remove_missing_level"] += 1
            return
        book[price] -= size
        if book[price] < 0:
            self.stats["negative_level"] += 1
            if self.strict:
                raise RuntimeError(f"negative level {side} {price}: {book[price]}")
            del book[price]                   
        elif book[price] == 0:
            del book[price]

    def add(self, order_id, side, price, size):
        self.orders[order_id] = {"side": side, "price": price, "size": size}
        self._add_to_level(side, price, size)

    def cancel(self, order_id, size):
        o = self.orders.get(order_id)
        if o is None:
            self.stats["unknown_cancel"] += 1
            return
        cut = min(size, o["size"])
        self._remove_from_level(o["side"], o["price"], cut)
        o["size"] -= cut
        if o["size"] == 0:
            del self.orders[order_id]

    def modify(self, order_id, side, price, size):
        o = self.orders.get(order_id)
        if o is None:
            self.stats["unknown_modify"] += 1
            self.add(order_id, side, price, size)
            return
        self._remove_from_level(o["side"], o["price"], o["size"])
        self.orders[order_id] = {"side": side, "price": price, "size": size}
        self._add_to_level(side, price, size)

    def clear(self):
        self.orders.clear()
        self.bids.clear()
        self.asks.clear()

    def best_bid(self):
        return self.bids.peekitem(-1) if self.bids else None

    def best_ask(self):
        return self.asks.peekitem(0) if self.asks else None

    def top_n_bids(self, n=10):
        return [self.bids.peekitem(-(i + 1)) for i in range(min(n, len(self.bids)))]

    def top_n_asks(self, n=10):
        return [self.asks.peekitem(i) for i in range(min(n, len(self.asks)))]

    def validate(self):
        expected = defaultdict(int)
        for o in self.orders.values():
            expected[(o["side"], o["price"])] += o["size"]
        actual = defaultdict(int)
        for p, s in self.bids.items():
            actual[("B", p)] = s
        for p, s in self.asks.items():
            actual[("A", p)] = s
        assert expected == actual, f"mismatch:\n expected {dict(expected)}\n actual {dict(actual)}"

if __name__ == "__main__":
    b = Book(strict=True)
    b.add(1, "B", 100.00, 5)
    b.add(2, "B", 99.75, 3)
    b.add(3, "A", 100.25, 4)
    b.validate()

    b.cancel(1, 2)
    b.validate()
    b.cancel(1, 3)
    b.validate()
    assert b.best_bid() == (99.75, 3)
    print("Test 1 passed")

    b = Book(strict=True)
    b.add(10, "A", 101.00, 6)
    b.modify(10, "A", 101.25, 6)
    b.validate()
    assert b.asks[101.25] == 6 and 101.00 not in b.asks
    print("Test 2 passed")

    b = Book(strict=True)
    b.add(20, "B", 99.50, 2)
    b.clear()
    assert b.best_bid() is None and not b.orders
    print("Test 3 passed")