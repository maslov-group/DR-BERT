# DR-BERT: A Protein Language Model to Annotate Disordered Regions

Despite their lack of a rigid structure, intrinsically disordered regions in proteins play important roles in cellular functions, including mediating protein-protein interactions. Therefore, it is important to computationally annotate disordered regions of proteins with high accuracy. Most popular tools use evolutionary or biophysical features to make predictions of disordered regions. In this study, we present DR-BERT, a compact protein language model that is first pretrained on a large number of unannotated proteins before being trained to predict disordered regions. Although it does not use any evolutionary or biophysical information, DR-BERT shows a statistically significant improvement when compared to several existing methods on a gold standard dataset. We show that this performance is due to the information learned during pre-training and DR-BERT's ability to use contextual information. A web application for using DR-BERT is available at https://huggingface.co/spaces/nambiar4/DR-BERT and you can also use this repo to also get DR-BERT scores. ![alt text](https://nambiar4.web.illinois.edu/src/rnap.gif "Logo Title Text 1")

## Required packages (to obtain DR-BERT scores)
* python3
* transformers
* pandas
* pytorch

If you do not already have these packages installed, we have found that these commands should get you the minimal conda environment
```shell
conda create --name test python==3.9
conda activate test
conda install -c huggingface transformers
conda install pandas
conda install pytorch
```
Alternatively, we also have made available a `minimal.yml` file that you can use to create the conda environment by saying
```shell
conda env create -f minimal.yml
```
You do not need a GPU for to get DR-BERT scores as long as it's not for too many sequences!

## Downloading the model checkpoints
You can download the DR-BERT model weights from our Google Drive folder [here](https://drive.google.com/drive/folders/1hMAnXaPrK9HPzcC0RZ5d7zsw3IXN910f?usp=sharing). Once you have downloaded the checkpoint, unzip it.

## Getting DR-BERT scores
Once you're all set up you can use the `get_scores_fasta` script like this
```shell
python3 get_scores_fasta.py ./DR-BERT-final/ sample.fa sample_out.pkl
 ```
 Where `./DR-BERT-final/` is the unzipped model weights folder you have downloaded, `sample.fa` is the fasta file with sequences that you want to get DR-BERT scores for and `sample_out.pkl` is the name of the output file.

## Training your own model
If you want to finetune (or pretrain) your own model, you can use the appropriate notebook from the Notebooks directory and download the relevant data [here](https://drive.google.com/drive/folders/1tCZc6nNpk9hATftKIw5QF4wfoGomV0wd?usp=sharing). To finetune, you should also start with the pretrained model weights found [here](https://drive.google.com/drive/folders/1hMAnXaPrK9HPzcC0RZ5d7zsw3IXN910f?usp=sharing).
