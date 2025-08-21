import struct

class CoreReportConfigEntry:
    # (field_name, type, init input)
    SLOTS = [
        ("report_name", str, True),
        ("target_name", str, True),
        ("report_type", str, True),
        ("report_variable", str, True),
        ("unit", str, True),
        ("report_format", str, True),
        ("target_type", str, True),
        ("dt", float, True),
        ("start_time", float, True),
        ("end_time", float, True),
        ("num_gids", int, False),  # computed dynamically
        ("buffer_size", int, True),
    ]

    def __init__(self, *args, **kwargs):
        inputs = [(name, typ) for name, typ, init_input in self.SLOTS if init_input]

        # fill positional args
        for (name, typ), value in zip(inputs, args):
            assert isinstance(value, typ), f"Expected {name} to be {typ.__name__}, got {type(value).__name__}"
            setattr(self, name, value)

        # fill remaining from kwargs
        for name, typ in inputs[len(args):]:
            if name not in kwargs:
                raise ValueError(f"Missing required argument: {name}")
            value = kwargs[name]
            assert isinstance(value, typ), f"Expected {name} to be {typ.__name__}, got {type(value).__name__}"
            setattr(self, name, value)

        self.gids = []

    @property
    def num_gids(self):
        return len(self.gids)

    def set_gids(self, gids):
        assert isinstance(gids, list), "gids must be a list"
        self.gids = gids

    def dump(self, f):
        if not self.gids:
            raise ValueError(f"Cannot dump entry: gids not set or empty: `{self.gids}`")

        # text line with num_gids in correct position
        line_values = []
        for name, _, _ in self.SLOTS:
            line_values.append(str(getattr(self, name)))
        f.write((" ".join(line_values) + "\n").encode())

        # binary gids
        f.write(struct.pack(f"{self.num_gids}i", *self.gids))
        f.write(b"\n")  # separator

    @classmethod
    def load_from_file(cls, f):
        # read text line
        line = f.readline()
        if not line:
            return None  # EOF

        tokens = line.decode().strip().split()
        if len(tokens) != len(cls.SLOTS):
            raise ValueError(f"Expected {len(cls.SLOTS)} fields, got {len(tokens)}")

        # convert tokens to proper types
        kwargs = {}
        for (name, typ, _), tok in zip(cls.SLOTS, tokens):
            if name == "num_gids":
                num_gids = int(tok)  # store to read binary
            else:
                kwargs[name] = typ(tok)

        entry = cls(**kwargs)

        # read binary gids
        gids_data = f.read(num_gids * 4)
        if len(gids_data) != num_gids * 4:
            raise ValueError(f"Expected {num_gids*4} bytes for gids, got {len(gids_data)}")

        gids = list(struct.unpack(f"{num_gids}i", gids_data))
        entry.set_gids(gids)

        # consume separator newline
        f.readline()

        return entry
    
    def __eq__(self, other):
        if not isinstance(other, CoreReportConfigEntry):
            return NotImplemented

        # compare all slots except num_gids (computed) and gids separately
        for name, _, init_input in self.SLOTS:
            if init_input:  # fields that were set in init
                if getattr(self, name) != getattr(other, name):
                    return False

        # compare gids
        return self.gids == other.gids

    def __repr__(self):
        vals = {name: getattr(self, name) if name != "num_gids" else self.num_gids for name, _, _ in self.SLOTS}
        vals["gids"] = self.gids
        return f"CoreReportConfigEntry({vals})"