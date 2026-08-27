# P275 source-transport prescore amendment

The frozen source, roles, object identities, limits, gates, stop rule and claim
ceiling are unchanged.

Two attempted formal controller starts produced no report and read no binary
body.  The first failed during the initial Python `urllib` TLS handshake.  The
second obtained the commit object and then failed during the recursive-tree TLS
handshake.  A diagnostic filtered Git fetch also failed during its initial
Schannel handshake and produced no source object.

Before any complete tree, text-object set, scientific payload or report existed,
P275 fixes only the network transport to installed `curl.exe`, HTTP/1.1, with at
most two retries for transport errors.  A bounded probe retrieved the exact
34,925-byte recursive-tree response and its frozen tree identity.  Formal runs
still request only the commit, tree and eight small text Git blobs; DNG and
checkpoint bodies remain unread.  No mirror, alternate source, object change,
timeout relaxation, scientific gate or interpretation changes.
