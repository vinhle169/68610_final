import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics_enc_lstm(original_data, combined_data):
    """
    :param original_data: path to model metrics for training on original data
    :param combined_data: path to model metrics for training on combined data
    :return:
    """
    def grab_loss_and_perplexity(line: str):
        idx = line.find(',')
        loss = float(line[line.find(':')+1: idx])
        perplexity = float(line[line.rfind(':')+1: ])
        return loss, perplexity

    with open(original_data, 'r') as f:
        org_metrics = f.read()
    with open(combined_data, 'r') as f:
        comb_metrics = f.read()

    org_metrics = org_metrics.split('\n')
    comb_metrics = comb_metrics.split('\n')

    org_metrics = [i[i.find(',')+1:] for i in org_metrics]
    comb_metrics = [i[i.find(',') + 1:] for i in comb_metrics]

    org_loss, org_perplexity = [], []
    comb_loss, comb_perplexity = [], []

    for i in range(len(org_metrics)):
        org_i = org_metrics[i]
        comb_i = comb_metrics[i]
        org_l, org_p = grab_loss_and_perplexity(org_i)
        comb_l, comb_p = grab_loss_and_perplexity(comb_i)
        org_loss.append(org_l)
        org_perplexity.append(org_p)
        comb_loss.append(comb_l)
        comb_perplexity.append(comb_p)

    x = list(range(1, len(org_metrics)+1))
    sns.set_style('darkgrid')

    plt.plot(x, org_loss)
    plt.plot(x, comb_loss)
    plt.legend(['Flickr Dataset', 'Combined Dataset'], loc='upper left')
    plt.title('Loss Over Time(Clip + GPT-2)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig('loss_fig_clip.png')
    plt.clf()

    plt.plot(x, org_perplexity)
    plt.plot(x, comb_perplexity)
    plt.legend(['Flickr Dataset', 'Combined Dataset'], loc='upper left')
    plt.title('Perplexity Over Time(Clip + GPT-2)')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')

    plt.savefig('perplexity_fig_clip.png')
    plt.clf()


if __name__ == '__main__':
    original_data = 'training_log_orig_clipgpt.txt'
    combined_data = 'training_log_clipgpt.txt'
    plot_metrics_enc_lstm(original_data, combined_data)



