# SF2.4R Newgrain source reconnaissance

Date: 2026-07-18

Node: `ULT > RF0.4 > SF2.4R`

Status: complete; terms-blocked before a formal metadata audit

## Question

Does a film-specific community platform expose a materially better exact-stock
and shared-photographer design than the closed Commons/YFCC pools, with terms
that permit a bounded metadata-only connectivity audit?

This is source reconnaissance, not a preregistered stock experiment. It cannot
establish stock use, author identity, current pixel rights, stock appearance or
identifiability.

## Authoritative and observed source facts

Newgrain's public product description says that the service is limited to
analogue photography and attaches controlled film-stock, format, camera, lab,
scanner, push/pull and development metadata to posts. Its public application
currently exposes a search-only stock catalogue.

A minimal anonymous reconnaissance performed before the terms decision found:

- 385 approved catalogue entries in one public search response;
- exact catalogue identities for Ektar 100, Velvia 50, UltraMax 400, Gold 200,
  Portra 400, Pro 400H and Ektachrome E100;
- catalogue-level post counts ranging from tens to several thousand for those
  examples;
- a two-document projection-only schema probe containing stock, pseudonymous
  user, format, lab, scanner, push/pull and developing-method fields.

Two catalogue queries and one two-document metadata query were made. No account
was created, no login was used, no image URL or image payload was requested,
and no response body or user identifier was retained in the repository. These
observations establish technical/source-design potential only and are forbidden
as confirmatory connectivity evidence.

Primary source checkpoints:

- https://www.newgrain.app/
- https://apps.apple.com/app/newgrain-film-photo-community/id6444198677
- https://newgrain.framer.website/terms-and-conditions
- https://newgrain.framer.website/privacy-policy

## Terms gate

The published Terms of Use, version 1.0 dated 2022-10-11, grant only a limited
personal, non-commercial access licence. More importantly, the Acceptable Use
Policy expressly prohibits automated agents or scripts that generate automated
searches, requests or queries, or strip, scrape or mine Site data. The public
frontend's technical ability to read catalogue or post metadata does not waive
that restriction. User-uploaded content also remains owned by its uploader;
Newgrain receives only an optional promotional-use grant.

The reconnaissance therefore stopped as soon as this governing restriction was
verified. Exposed frontend configuration is not permission, and it must not be
copied into project configs or treated as a reusable credential.

## Decision

`terms_blocked_no_formal_audit_dor`.

Do not implement a Newgrain client, enumerate posts/users, compute a shared-user
graph, retain metadata snapshots, request images or use this source for fitting,
training or latent-mode work. A future branch requires written platform
permission or an officially published research/data export whose terms cover
the intended audit and downstream pixel use. Seeking such permission is an
external communication and is outside the current autonomous authority.

This decision does not say that the source lacks useful stock connectivity. It
says that the project cannot lawfully and reproducibly test that hypothesis by
automated acquisition under the currently published terms.

## Change propagation and claim boundary

- parent `RF0.4` gains a high-support but terms-blocked source-design result;
- sibling Commons/YFCC/NASA/Openverse/Smithsonian conclusions remain unchanged;
- `LSM1` stays ineligible because there is no allowed manifest, rights-cleared
  pixel pool or stock-identifiability evidence;
- no stock evidence grade, operator, training, clustering, `S1/S2`, calibration,
  authenticity or product claim opens;
- Ultimate remains active and returns to another independently permitted source
  design or a deterministic product leaf without weakening any frozen gate.
