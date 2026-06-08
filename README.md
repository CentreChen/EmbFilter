# EmbFilter 
<!--[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20507711.svg)](https://doi.org/10.5281/zenodo.20507711)-->
Official Implementation for paper "Your Embedding Matrix is secretly a Feature Lens for Text Embeddings".
This repository introduces a simple, lightweight linear filter designed to refine zero-shot text embeddings.

### tips
- recommend: python 3.10, torch==2.6.0, mteb==1.4.0, transformers==4.52.3
- if fail to load `SickrSTS`, change path `MMathematica/sickr-sts` in the mteb package (*/mteb/tasks/STS/en/SickrSTS.py) to `mteb/sickr-sts`
- if fail to load `MindSmallReranking`, try datasets==2.18 by `pip install datasets==2.18`

### run
- run the EmbFilter with `python run4qwen_prompteol.py --filter_ratio 2`
- `filter_ratio` is the ratio of dims to be saved, e.g., `filter_ratio=1` means saving 1/1=100% dims, `filter_ratio=2` means saving 1/2=50% dims, and so on.

# Reference
This paper has informed us a new design for LLM text embedding training, which stays tuned for the release!
If you find this code useful useful for your research, please cite our paper.
```
@misc{wu2026unembeddingmatrixsecretlyfeature,
      title={Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings}, 
      author={Songhao Wu and Zhongxin Chen and Yuxuan Liu and Heng Cui and Cong Li and Rui Yan},
      year={2026},
      eprint={2606.07502},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.07502}, 
}
```

