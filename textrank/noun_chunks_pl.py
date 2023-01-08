# © https://github.com/explosion/spaCy/discussions/7006
def is_noun_phrase_root(word, np_deps, conj):
    if word.dep in np_deps:
        return True
    elif word.dep == conj: # two elements connected by a coordinating conjunction, such as and, or, etc
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
        "iobj",  # indirect object
        "obj",
        "obl",
        "obl:arg",
    ]
    mod_labels = [ # modifirts
        "amod",  # adjectival modifier -- adjectival phrase that modifies the meaning of the NP eg. Sam eats red **meat**.
        "nmod"  # nominal modifiers
    ]
    doc = doclike.doc

    if not doc.has_annotation("DEP"): # Dep: Syntactic dependency, i.e. the relation between tokens.
        raise ValueError(Errors.E029)

    np_deps = [doc.vocab.strings.add(label) for label in labels] # strings shared across multiple documents. It's a lookup table in both directions (string <-> its hash)
    conj = doc.vocab.strings.add("conj")
    mod_deps = [doc.vocab.strings.add(label) for label in mod_labels]
    np_label = doc.vocab.strings.add("NP")
    prev_end = 0
    for i, word in enumerate(doclike):
        if word.pos_ not in ("NOUN", "PROPN"):  # TODO PRONs (nazwy własne) are mostly stop words, should I include it?
            continue
        if is_noun_phrase_root(word, np_deps, conj):
            start = word.i
            end = start + 1
            while start > 0 and start >= prev_end and doc[start-1].head in [doc[start], word] and doc[start-1].dep in mod_deps:
                start -= 1
            while end < len(doc) and doc[end].head in [doc[end-1], word] and doc[end].dep in mod_deps:
                end += 1
            prev_end = end
            yield start, end, np_label
