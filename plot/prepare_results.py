"""Regenerate paper tables/figures: python plot/prepare_results.py --output-dir PATH."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, default=REPO, help="Root directory of experiment results.")
parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "figures",
                    help="Directory for generated paper figures and tables.")
args = parser.parse_args()
ROOT = args.input_dir.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)

def one_input(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if not matches:
        # Fresh runs use plain directories; historical downloads used dated folders.
        matches = sorted(directory.glob(pattern.replace('-*/*/', '/')))
    if len(matches) != 1:
        raise ValueError(f"Expected one input for {directory / pattern}, found {len(matches)}")
    return matches[0].relative_to(ROOT)

def read(key, path):
    path = ROOT / path
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != 'checkpoint'}
    return d
def cell(m, s):
    return f'${m:.1f} \\pm {s:.1f}$'
def table(name, caption, headers, rows, label):
    if name in {'main_results', 'main_results_N40', 'main_results_N100'}:
        caption += ' Bold indicates the highest mean return among non-oracle policies.'
        for col in range(1, len(headers)):
            eligible = [row for row in rows if row[0] != 'Oracle lookahead']
            best = max(float(row[col].strip('$').split()[0]) for row in eligible)
            for row in eligible:
                if float(row[col].strip('$').split()[0]) == best:
                    row[col] = '$\\boldsymbol{' + row[col].strip('$') + '}$'
    text = '\\begin{table}[H]\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\centering\n\\small\n'
    text += '\\begin{tabular}{l' + 'c'*(len(headers)-1) + '}\n\\toprule\n'
    text += ' & '.join(headers) + ' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(row) + ' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (OUT / (name+'.tex')).write_text(text, encoding='utf-8')

envs = ['Stationary', 'Drifting', 'MIMIC-ICU']
dirs = ['synthetic-stationary', 'synthetic-drifting', 'mimic-icu']
main = [read('main_'+e, 'experiment_outputs/main_comparison/'+f+'_comparison.npz') for e,f in zip(envs,['synthetic_stationary','synthetic_drifting','mimic_icu'])]
labels = {'random':'Random','obs_greedy':'Observation-greedy','activation_greedy':'Activation-greedy','neurwin':'NeurWIN','ppo':'PPO','mlp_actor':'Deterministic MLP','gaussian_actor':'Gaussian actor','learned_rollout':'Learned rollout','dpmd':'BIRD','oracle_lookahead':'Oracle lookahead'}
selected = ['random', 'obs_greedy', 'activation_greedy', 'neurwin', 'ppo', 'learned_rollout', 'dpmd', 'oracle_lookahead']
def select_methods(d):
    idx = [list(d['method_names']).index(n) for n in selected]
    return dict(d, method_names=np.asarray(selected),
                mean_returns=d['mean_returns'][idx], std_returns=d['std_returns'][idx])
main = [select_methods(d) for d in main]
rows=[]
for i, method in enumerate(selected):
    rows.append([labels[method]]+[cell(d['mean_returns'][i],d['std_returns'][i]) for d in main])
table('main_results','Expanded main comparison at $N=20,K=5$: mean return $\\pm$ episode standard deviation over 100 episodes of length 100. Oracle lookahead uses privileged information.', ['Policy']+envs, rows,'tab:expanded-main')
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig, axes=plt.subplots(1,3,figsize=(5.5,2.65), layout='constrained')
for ax,e,d in zip(axes,envs,main):
    names=d['method_names']; colors=['#167d9a' if n=='dpmd' else '#aaaaaa' if n=='oracle_lookahead' else '#d4dfe7' for n in names]
    ax.barh(range(len(names)),d['mean_returns'],xerr=d['std_returns'],color=colors,error_kw={'elinewidth':0.65,'capsize':1.5})
    ax.set_yticks(range(len(names)),[labels[n] for n in names] if ax is axes[0] else ['']*len(names),fontsize=7)
    ax.invert_yaxis(); ax.set_title(e); ax.set_xlabel('Episode return'); ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'expanded_main.pdf'); plt.close(fig)

for n, k in [(40, 10), (100, 25)]:
    scale = [select_methods(read('scale', f'{dr}/experiment_outputs/all_policies_transfer/comparison_N{n}_K{k}.npz')) for dr in dirs]
    rows = [[labels[name]] + [cell(d['mean_returns'][i], d['std_returns'][i]) for d in scale] for i, name in enumerate(selected)]
    caption = (f'Main comparison at $N={n},K={k}$: mean return $\\pm$ episode standard deviation over 100 episodes of length 100. '
               'Learned policies reuse the main-comparison checkpoints trained at $N=20,K=5$ without retraining. Oracle lookahead uses privileged information.')
    table(f'main_results_N{n}', caption, ['Policy']+envs, rows, f'tab:main-N{n}')

collections={}
for kind, pattern in [('aux','experiment_outputs/aux_loss_ablation/*.npz'),('critic','experiment_outputs/critic_ablation-*/*/*.npz'),('encoder','encoder_ablation_N20_K5.npz'),('factor','experiment_outputs/factor_stress-*/*/*.npz')]:
    collections[kind]=[read(kind+'_'+e, one_input(ROOT/dr, pattern)) for dr,e in zip(dirs,envs)]
for kind, names, caption in [
    ('aux',['With auxiliary loss','Without auxiliary loss'],'Auxiliary prediction loss ablation.'),
    ('critic',['Per-arm twin','Per-arm single','Joint twin'],'Critic ablation; these independently trained checkpoints differ from the main-comparison checkpoints.'),
    ('encoder',['Transformer','LSTM','MLP'],'History encoder ablation ($L=40$ synthetic, $L=80$ MIMIC-ICU).')]:
    rows=[[name]+[cell(d['mean_returns'].reshape(-1)[i],d['std_returns'].reshape(-1)[i]) for d in collections[kind]] for i,name in enumerate(names)]
    table(kind+'_results',caption+' Mean return $\\pm$ episode standard deviation; one training seed and 100 evaluation episodes per variant.', ['Variant']+envs,rows,'tab:'+kind)
rows=[]
for setting in range(2):
    for i,name in enumerate(['Observation-greedy','Activation-greedy','Learned rollout','BIRD']):
        rows.append([('Known / ' if setting==0 else 'Unknown / ')+name]+[cell(d['mean_returns'][setting,i],d['std_returns'][setting,i]) for d in collections['factor']])
table('factor_results','Factor stress tests. Known: noisy observations with fixed homogeneous dynamics. Unknown: noiseless observations with hidden heterogeneous dynamics (noiseless vitals in MIMIC-ICU). $L=40$, $N=20,K=5$; 100 evaluation episodes, one training seed.', ['Setting / policy']+envs,rows,'tab:factor')
fig,axes=plt.subplots(1,3,figsize=(5.5,2.4),layout='constrained')
for ax,e,j in zip(axes,envs,range(3)):
    vals=[collections['aux'][j]['mean_returns'][1]/collections['aux'][j]['mean_returns'][0], collections['critic'][j]['mean_returns'][1]/collections['critic'][j]['mean_returns'][0],collections['critic'][j]['mean_returns'][2]/collections['critic'][j]['mean_returns'][0],collections['encoder'][j]['mean_returns'][1,0]/collections['encoder'][j]['mean_returns'][0,0],collections['encoder'][j]['mean_returns'][2,0]/collections['encoder'][j]['mean_returns'][0,0]]
    ax.barh(range(5),vals,color='#167d9a'); ax.set_yticks(range(5),['No auxiliary loss','Single critic','Joint critic','LSTM encoder','MLP encoder'] if j==0 else ['']*5,fontsize=7); ax.axvline(1,color='black',ls='--',lw=1);ax.set_xlim(0,1.2);ax.set_title(e);ax.set_xlabel('Relative return');ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'component_ablations.pdf'); plt.close(fig)

ws=read('window_stationary',dirs[0]+'/experiment_outputs/window_l_ablation/window_l_ablation_N20_K5.npz')
ws1=read('window_stationary_seed1',dirs[0]+'/experiment_outputs/window_l_ablation/window_l_ablation_N20_K5_seed1.npz')
wd=read('window_drifting',one_input(ROOT/dirs[1], 'experiment_outputs/window_l_ablation-*/*/*.npz'))
wm=[read('window_mimic_'+str(l), dirs[2]+f'/experiment_outputs/window_l_ablation/mimic_L{l}_main_eval.npz') for l in [40,80]]
rows=[]
for i,l in enumerate([20,40,80]):
    m='---' if l==20 else cell(wm[i-1]['mean_returns'][8],wm[i-1]['std_returns'][8])
    rows.append([str(l),cell(ws['mean_returns'][0,i],ws['std_returns'][0,i]),cell(wd['mean_returns'][0,i],wd['std_returns'][0,i]),m])
table('window_results','History-window sensitivity. Mean return $\\pm$ episode standard deviation over 100 episodes; separate checkpoints from the main comparison. No $L=20$ MIMIC result is reported.', ['$L$']+envs,rows,'tab:window')
markov = read('markov2', 'markov2/results_50_10_known_activation.npz')
random_mean = markov['random'].mean()
whittle_gap = markov['true_whittle'].mean() - random_mean
if np.isclose(whittle_gap, 0):
    raise ValueError('Cannot report Markov2 gap closure: Whittle and random means coincide.')
rows = []
for method, label in [('random', 'Random'), ('obs_greedy', 'Observation-greedy'),
                      ('activation_greedy', 'Activation-greedy'), ('ppo', 'PPO'),
                      ('neurwin', 'NeurWIN'), ('dpmd', 'BIRD'),
                      ('true_whittle', 'True Whittle (oracle)')]:
    returns = markov[method]
    gap = 100 * (returns.mean() - random_mean) / whittle_gap
    rows.append([label, cell(returns.mean(), returns.std()), f'{gap:.1f}' + r'\%'])
table('markov2_results', 'Fully observable two-state Markov RMAB ($N=50,K=10$): '
      '100 evaluation episodes. Mean return and episode standard deviation; '
      'gap captured is relative to random and true Whittle. Activation-greedy '
      'uses the true transition probabilities.',
      ['Policy', 'Return', 'Whittle gap captured'], rows, 'tab:markov2')
print('Generated evidence-backed tables and figures.')
