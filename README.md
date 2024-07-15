# Poirot
Provide retrieval to heracles

# start

Install conda  [Miniconda](https://docs.anaconda.com/miniconda/)

```sh
# Install dependency
conda create -n poirot python=3.11
conda activate poirot

pip install poetry

# Prepare for env
cp .env.example .env

# Rush!
python utils/ticket_utils.py 
```
