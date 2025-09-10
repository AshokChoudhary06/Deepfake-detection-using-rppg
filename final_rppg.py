import pyVHR as vhr 
import numpy as np 
from pyVHR.BVP import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from scipy.signal import welch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
from sklearn.model_selection import GroupShuffleSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVC
from scipy.signal import coherence, welch
from sklearn.metrics import mutual_info_score
from scipy.stats import skew, kurtosis
import antropy as ant
import nolds
import os

class VideoProcessor:
    """Processes videos to extract BVP signals"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.fps = None
        self.filtered_patch_bvps = None  # Shape: (windows, patches, time_points)
        self.windowed_patch_sig = None
        self.filtered_windowed_patch_sig = None
        
    def process(self):
        """Full rPPG processing pipeline"""
        try:
            # 1. Get FPS
            self.fps = vhr.extraction.get_fps(self.video_path)
            
            # 2. Configure signal extraction
            sig_extract = vhr.extraction.SignalProcessing()
            sig_extract.choose_cuda_device(0)
            sig_extract.set_skin_extractor(vhr.extraction.SkinExtractionConvexHull('GPU'))
            vhr.extraction.SkinProcessingParams.RGB_LOW_TH =  0
            vhr.extraction.SkinProcessingParams.RGB_HIGH_TH = 240
            # 3. Define facial landmarks
            landmarks = vhr.extraction.MagicLandmarks.cheek_left_top +\
            vhr.extraction.MagicLandmarks.forehead_center +\
            vhr.extraction.MagicLandmarks.forehoead_right +\
            vhr.extraction.MagicLandmarks.cheek_right_top +\
            vhr.extraction.MagicLandmarks.forehead_left +\
            vhr.extraction.MagicLandmarks.nose 
            sig_extract.set_landmarks(landmarks)
            
            # 4. Extract patches
            sig_extract.set_square_patches_side(80.0)
            patch_sig = sig_extract.extract_patches(self.video_path, "squares", "mean")
            
            # 5. Window signals
            wsize = 8  # 8-second windows
            self.windowed_patch_sig, timeES = vhr.extraction.sig_windowing(patch_sig, wsize, 1, self.fps)
            
            # 6. Filter and convert to BVP
            self.filtered_windowed_patch_sig = vhr.BVP.apply_filter(self.windowed_patch_sig, vhr.BVP.rgb_filter_th, params = {'RGB_LOW_TH':0, 'RGB_HIGH_TH':230})
            
            patch_bvps = RGB_sig_to_BVP(self.filtered_windowed_patch_sig, self.fps,device_type ='cuda', method = cupy_CHROM)
            
            self.filtered_patch_bvps = vhr.BVP.apply_filter(patch_bvps, BPfilter, params = {'order':6, 'minHz':0.75, 'maxHz':4.0, 'fps':self.fps})
            return True
        except Exception as e:
            print(f"Error processing {os.path.basename(self.video_path)}: {str(e)}")
            return False

    def video_visualize(self):
        return vhr.plot.display_video(self.video_path)

    def visualize_windowed_signal(self):
        """ Helps in visualizing the RGB signals without filter"""
        w = np.random.randint(0, len(self.windowed_patch_sig))
        vhr.plot.visualize_windowed_sig(self.windowed_patch_sig, w)
    
    def visualize_filtered_windowed_sig(self):
        w = np.random.randint(0, len(self.filtered_windowed_patch_sig))
        vhr.plot.visualize_windowed_sig(self.filtered_windowed_patch_sig, w)
        

    def visualize_bvps(self):
        w = np.random.randint(0,len(self.filtered_patch_bvps))
        vhr.plot.visualize_BVPs(self.filtered_patch_bvps,w)

    def visualize_bpm(self):
        #patch_bpmes = vhr.BPM.BVP_to_BPM_cuda(self.filtered_patch_bvps, self.fps)
        vhr.plot.visualize_BVPs(self.filtered_patch_bvps, 1)

class FeatureEngine:
    """Extracts features from BVP signals as per paper"""
    def __init__(self, bvp_data):
        """
        Args:
            bvp_data: Output from VideoProcessor.filtered_patch_bvps 
                      Shape: (windows, patches, time_points)
        """
        self.bvp = bvp_data
        self.feature_df = None
        
    def create_features(self):
        """Main feature extraction workflow"""
        features = []
        
        # Process each window
        for window in self.bvp:
            # 1. Intra-patch features (12)
            intra = self._get_intra_features(window)
            
            # 2. Inter-patch features (8)
            inter = self._get_inter_features(window)
            combined = np.concatenate([intra, inter])  # Shape: (20,)
            features.append(combined)
            
        # Create DataFrame
        columns = [
            # Intra features
            'ZCR', 'Mobility', 'Complexity', 'SpectralEntropy',
            'SampleEntropy', 'PetrosianFD', 'HiguchiFD', 'KatzFD', 
            'DFA', 'SVDAvg',
            # Inter feature
            'MIMean', 'MIStd'
        ]
        
        self.feature_df = pd.DataFrame(features, columns=columns)
        return self.feature_df

    def _get_intra_features(self, window):
        """12 intra-patch complexity features"""
        feats = []
        for patch in window:
            # Time-domain

            patch = np.asarray(patch).flatten()
            zcr = len(np.where(np.diff(np.sign(patch)))[0]) / len(patch)
            mobility, complexity = ant.hjorth_params(patch)
            
            # Frequency-domain
            _a, psd = welch(patch, fs=30)
            psd_norm = psd / psd.sum()
            spectral_ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            # Entropy
            sampen = ant.sample_entropy(patch)
            
            # Fractal dimensions
            p_fd = ant.petrosian_fd(patch)
            higuchi = ant.higuchi_fd(patch)
            katz = ant.katz_fd(patch)
            dfa = nolds.dfa(patch)
            
            # SVD-based
            U, s, vt = np.linalg.svd(patch.reshape(-1, 1), full_matrices=False)
            svd_avg = np.mean(s)
            #svd_skew = skew(s)
            #svd_flatness = np.std(s) / np.mean(s)
            
            feats.append([zcr, mobility, complexity, spectral_ent, sampen,
                         p_fd, higuchi, katz, dfa, svd_avg])
        
        return np.mean(feats, axis=0)

    def _get_inter_features(self, window):
        """8 inter-patch coherence features"""
        spectral_sims, mi_scores = [], []
        num_patches = window.shape[0]
        
        # Spectral coherence
        """for i in range(num_patches):
            for j in range(i+1, num_patches):
                f, Cxy = coherence(window[i], window[j], fs=30)
                spectral_sims.append(np.mean(Cxy))"""
        
        # Mutual information
        for i in range(num_patches):
            for j in range(i+1, num_patches):
                mi = mutual_info_score(window[i], window[j])
                mi_scores.append(mi)
        
        # Compute moments
        return [ 
            np.mean(mi_scores), np.std(mi_scores)
        ]

    def save_features(self, filename):
        """Save features to CSV"""
        if self.feature_df is not None:
            self.feature_df.to_csv(filename, index=False)
        else:
            raise ValueError("Run create_features() first")