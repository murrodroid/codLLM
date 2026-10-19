# Licensing and citation

codLLM permits commercial and non-commercial use, modification, and redistribution under standard licenses.
The license depends on the material; these are not alternative licenses for the same material.

| Material | License |
|---|---|
| Project software, scripts, tests, configuration, and code examples, including code in documentation | [Apache-2.0](../LICENSE) |
| Project-authored prose, documentation, figures, organization/model cards, and the authors' contributions to `codLLM_Thesis.pdf` | [CC BY 4.0](../LICENSES/CC-BY-4.0.txt) |
| codLLM model contributions released under this policy | [CC BY 4.0](../LICENSES/CC-BY-4.0.txt), with upstream model terms and notices retained |
| Third-party datasets, ICD10h masterlists/reference materials, dependencies, and attributed third-party content | Their respective terms; not relicensed by codLLM |

Unless a file or release explicitly states otherwise, this scope applies to project-owned materials from this update
onward. It does not grant rights that the authors do not hold. Third-party quotations, figures, and other attributed
material in the thesis are excluded from the CC BY grant unless their own notices say otherwise. Raw archival datasets
remain subject to their data-use agreements and are not distributed under either project license.

The license texts are authoritative. [NOTICE](../NOTICE) records authorship and upstream attribution;
[CITATION.cff](../CITATION.cff) provides the preferred citation.

## Attribution and citation

**Please credit Lucas E. B. Mortensen and Elias Dahl Lunøe when using codLLM, and cite the thesis in research publications.**
Also identify the model name and revision used, so that results can be traced to a specific checkpoint.

> Mortensen, Lucas E. B., and Elias Dahl Lunøe. 2026. *Automated Historical Cause-of-Death Coding Using Fine-Tuned
> Language Models*. Bachelor's thesis, DTU Compute, Technical University of Denmark.

[Read the thesis](https://huggingface.co/spaces/codllm/README/resolve/main/codLLM_Thesis.pdf).

When publicly sharing CC BY material or adaptations, give appropriate credit, link to the material and license,
retain supplied notices, and indicate changes as required by CC BY 4.0. For example:

> Based on codLLM by Lucas E. B. Mortensen and Elias Dahl Lunøe (2026),
> https://huggingface.co/codllm. Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.
> Changes: [describe your changes, or state that none were made].

For a model redistribution, link to the specific model and retain its upstream notices as well.
For software redistribution, follow Apache-2.0's license, notice, attribution, and change-marking requirements.

The research citation request is not an added license condition. These standard licenses do not impose a universal
citation obligation merely for running the model, operating a service, or publishing outputs that do not themselves
share licensed material. There are no additional non-commercial, field-of-use, or share-alike restrictions.

## Model provenance and prior releases

`codllm/codllm-base-deploy` is derived from Google's
[FLAN-T5-base](https://huggingface.co/google/flan-t5-base), distributed under Apache-2.0.
The CC BY declaration covers the codLLM contributions; it does not replace Google's license or attribution.
Model releases must carry the applicable upstream license text and notices alongside the codLLM license and citation.

Earlier software versions were released under MIT. The original notice is preserved in
[LICENSES/MIT-legacy.txt](../LICENSES/MIT-legacy.txt) and must be retained for incorporated MIT material.
Earlier model revisions were released under Apache-2.0. This policy does not revoke permissions already granted:
recipients can continue to use those earlier releases under their original terms, including where the same material
is also offered under the new policy. No retroactive attribution or citation requirement is asserted.

Future publication models must identify their actual upstream model, license scope, authors, and associated paper.
Do not use a planned publication as if it were an existing citation.

## Official license references

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)
- [Creative Commons attribution guidance](https://creativecommons.org/faq/#do-i-always-have-to-attribute-the-creator-of-the-licensed-material)

Creative Commons [recommends software-specific licenses for code](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software),
which is why codLLM software uses Apache-2.0 and its other project-owned materials use CC BY 4.0.
