# Poirot
Provide retrieval to heracles

# started

install conda  [Miniconda](https://docs.anaconda.com/miniconda/)

```sh
# Installation dependency
conda create -n poirot python=3.11
conda activate poirot

pip install poetry

# Prepare for configuration
cp .env.example .env

# Rush!
python utils/ticket_utils.py 
```
