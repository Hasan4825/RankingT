from pytracking.evaluation.environment import EnvSettings
import os

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/run/media/hasan/B709-D3C0/Hasan/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/run/media/hasan/B709-D3C0/Hasan/LaSOTBenchmark'
    # settings.network_path = '/run/media/mlcv/DATA/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.network_path = os.path.join(os.path.dirname(__file__), '..') + '/networks/'    # Where tracking networks are stored.
    settings.nfs_path = '/run/media/hasan/B709-D3C0/Hasan/toolkit/data/NfSDataset'
    settings.otb_path = '/run/media/hasan/B709-D3C0/Hasan/toolkit/data/OTB'
    settings.result_plot_path = '/run/media/hasan/B709-D3C0/Hasan/pytracking/pytracking/result_plots/'
    settings.results_path = '/run/media/hasan/B709-D3C0/Hasan/pytrackingRanking/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/run/media/hasan/B709-D3C0/Hasan/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = '/run/media/hasan/B709-D3C0/Hasan/toolkit/data/Temple-color-128'
    settings.trackingnet_path = '/run/media/hasan/B709-D3C0/Hasan/TrackingNet-devkit/TrackingNet'
    settings.uav_path = '/run/media/hasan/B709-D3C0/Hasan/toolkit/data/UAV123'
    settings.vot_path = '/run/media/hasan/B709-D3C0/Hasan/pytracking/pytracking/vot-workspace/sequences'
    settings.youtubevos_dir = ''

    return settings

