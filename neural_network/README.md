## Neural Network Write-up

Link to your dataset of 500 instances (or separate links to training/validation/testing
sets). (You can use Vanderbilt Box or Dropbox, etc., to share links.)

https://github.com/adamMgross/ai-art/tree/master/scraper/examples


PDF of project report that includes the following elements:
* Description of your dataset, how you have collected it, and how you have split it
into train/validation/test sets. (Note, some of the material here can be repeated
from your Assignment 3 report; think of this Assignment 4 report as a standalone
document, and include enough information that a new reader unfamiliar with your project would understand what you are doing, and why.)

Our dataset includes 675 images from artsy.net, a site dedicated to providing information and famous content from all styles of visual art. We are specifically interested in Surrealist and Impressionist paintings, so our images are about half surrealist and half impressionist. They all come from the most famous artists from those movements. We download these with a simple web-scraper that scrapes the top images from each artist’s page that we specify to the scraper. Thus we can add more URLs for more artists’ pages to download additional instances easily.

* Details of neural network experiments on MNIST, as well as neural network
experiments on your dataset. For each experiment, you should include:

i. What were the hyperparameter settings that stayed fixed in the
experiment?


ii. What was the hyperparameter that changed (the independent variable)?


iii. How long did it take to run the experiment, and what were the hardware
specs of the machine you were running it on?


iv. Show a plot of the training accuracy, validation accuracy, and testing
accuracy as a function of training epoch (these are the dependent
variables).

v. Your interpretation of what this plot means for each particular experiment.
Why do you think the results look the way they do?

