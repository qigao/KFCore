#include "kfcore/relation/predicate_routing_gate.hpp"

#include "tinytest.hpp"

#include <cstddef>
#include <string>

using namespace kfcore::relation;

spec("predicate routing gate public contract")
{
    it("uses the Apache reference dimensions")
    {
        PredicateRoutingGateOptions options;
        check(options.embedding_dim == std::size_t{512U});
        check(options.max_predicates == std::size_t{19103U});
        check(options.max_tensor_bytes > std::size_t{0U});
        check(options.max_vocabulary_bytes > std::size_t{0U});
    }

    it("names the routing model type explicitly")
    {
        check(
            kPredicateRoutingGateModelType ==
            "relation.predicate-routing-gate");
    }

    it("keeps routing provenance on PredicateVocabulary")
    {
        PredicateVocabulary vocabulary;
        vocabulary.routing_gate_provenance =
            "relation-gate:test";
        check(
            vocabulary.routing_gate_provenance ==
            "relation-gate:test");
    }
}
