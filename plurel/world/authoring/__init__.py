"""Authoring tools for a catalog of world schemas, one per step of the protocol in ``GUIDE.md``. A
:class:`~plurel.world.authoring.workspace.Workspace` says where a catalog lives; ``orders`` renders one
self-contained prompt per batch module of a sector's plan; ``realism`` is the gate on a new file and the census
over a catalog; ``dedupe`` finds near-duplicate records; ``review`` renders one reviewer prompt per file and one
fix prompt per file's findings; ``knowledge_orders`` does the same for a schema's world knowledge."""
