INSID3 is a **git submodule** pinned to a specific commit (reproducible eval).

```bash
# New clone of this repo
git clone --recurse-submodules <this-repo-url>

# Already cloned without submodules
git submodule update --init
```

That checks out `third_party/INSID3`. DINOv3 weights stay in `pretrain/` (gitignored; not part of the submodule).
