# © https://github.com/explosion/spaCy/discussions/7006
def is_np_root(word, np_deps, conj):
    if word.dep in np_deps:
        return True
    elif word.dep == conj:
        head = word.head
        while head.dep == conj and head.head.i < head.i:
            head = head.head
        return head.dep in np_deps
    else:
        return False


def noun_chunks_pl(doclike):
    labels = [
        "ROOT",
        "nsubj",
        "appos",
        "nsubjpass",
        "iobj",
        "obj",
        "obl",
        "obl:arg",
    ]
    mod_labels = [
        "amod",
        "nmod"
    ]
    doc = doclike.doc

    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)

    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    mod_deps = [doc.vocab.strings.add(label) for label in mod_labels]
    np_label = doc.vocab.strings.add("NP")
    prev_end = 0
    for i, word in enumerate(doclike):
        if word.pos_ not in ("NOUN", "PROPN"):  # TODO PRONs are mostly stop words, should I include it?
            continue
        if is_np_root(word, np_deps, conj):
            start = word.i
            end = start + 1
            while start >= prev_end and doc[start-1].head in [doc[start], word] and doc[start-1].dep in mod_deps:
                start -= 1
            while doc[end-1].head in [doc[end-1], word] and doc[end].dep in mod_deps:
                end += 1
            prev_end = end
            yield start, end, np_label
