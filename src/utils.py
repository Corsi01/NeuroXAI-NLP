import pandas as pd
import numpy as np
import os
import ipywidgets as widgets
from IPython.display import display
import h5py
import nibabel as nib
from nilearn import plotting, datasets
from IPython.display import display as ipy_display, clear_output
import glob

def load_transcript(transcript_path):
    """
    Loads a transcript file and returns it as a DataFrame.

    Parameters
    ----------
    transcript_path : str
        Path to the .tsv transcript file.

    """
    df = pd.read_csv(transcript_path, sep='\t')
    return df


def display_transcript(chunk_index, transcript_df):
    """
    Displays transcript text for a selected chunk.

    Parameters
    ----------
    chunk_index : int
        Index of the selected chunk.
    transcript_df : DataFrame
        DataFrame containing transcript data.

    """
    # Get the corresponding transcript row if it exists in the DataFrame
    transcript_chunk = transcript_df.iloc[chunk_index] if chunk_index < len(transcript_df) else None

    # Display the stimulus chunk number
    print(f"\n{'='*60}")
    print(f"Chunk number: {chunk_index + 1}")
    print(f"{'='*60}")

    # Display transcript details if available; otherwise, indicate no dialogue
    if transcript_chunk is not None and pd.notna(transcript_chunk['text_per_tr']):
        print(f"\nText: {transcript_chunk['text_per_tr']}")
        print(f"\nWords: {transcript_chunk['words_per_tr']}")
    else:
        print("\n<No dialogue in this scene>")


def create_dropdown_by_text(transcript_df):
    """
    Creates a dropdown widget for selecting chunks by their text.

    Parameters
    ----------
    transcript_df : DataFrame
        DataFrame containing transcript data.

    """
    options = []

    # Iterate over each row in the transcript DataFrame
    for i, row in transcript_df.iterrows():
        if pd.notna(row['text_per_tr']):  # Check if the transcript text is not NaN
            options.append((row['text_per_tr'], i))
        else:
            options.append(("<No dialogue in this scene>", i))
    return widgets.Dropdown(options=options, description='Select scene:')


def plot_fmri_on_brain(chunk_index, fmri_file_path, atlas_path, dataset_name, hrf_delay):
    """
    Map fMRI responses to brain parcels and plot it on a glass brain.

    Parameters
    ----------
    chunk_index : int
        The selected chunk from the transcript, used to determine the fMRI sample.
    fmri_file_path : str
        Path to the HDF5 file containing fMRI data.
    atlas_path : str
        Path to the atlas NIfTI file.
    dataset_name : str
        Name of the dataset inside the HDF5 file.
    hrf_delay : int
        Delay (in TR) between stimulus chunk and fMRI sample.
    """
    print(f"\n{'='*60}")
    print(f"fMRI BRAIN ACTIVITY")
    print(f"{'='*60}")

    # Load the atlas image
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    # Open the fMRI responses file, and extract the specific dataset
    with h5py.File(fmri_file_path, 'r') as f:
        fmri_data = f[dataset_name][()]
        print(f"fMRI dataset shape: {fmri_data.shape}")


    # Extract the corresponding sample from the fMRI responses
    if (chunk_index + hrf_delay) >= len(fmri_data):
        selected_sample = len(fmri_data) - 1
    else:
        selected_sample = chunk_index + hrf_delay

    fmri_sample_data = fmri_data[selected_sample]
    print(f"Extracting fMRI sample {selected_sample + 1} "
          f"(chunk {chunk_index + 1} + HRF delay {hrf_delay}).")

    # Map fMRI sample values to the brain parcels in the atlas
    output_data = np.zeros_like(atlas_data)
    for parcel_index in range(1000):
        output_data[atlas_data == (parcel_index + 1)] = fmri_sample_data[parcel_index]

    # Create the output NIfTI image
    output_img = nib.Nifti1Image(output_data, affine=atlas_img.affine)

   
    # Plot the glass brain with the mapped fMRI data
    display = plotting.plot_glass_brain(
        output_img,
        display_mode='lyrz',
        cmap='inferno',
        colorbar=True,
        plot_abs=False,
        threshold=1e-6,
        black_bg=False,   
    )

    plotting.show()


def interface_display_transcript_and_brain(transcript_path, fmri_file_path, 
                                          atlas_path, dataset_name, hrf_delay):
    """
    Interactive interface to display transcript chunks along with
    the fMRI response from the corresponding sample.

    Parameters
    ----------
    transcript_path : str
        Path to the .tsv transcript file.
    fmri_file_path : str
        Path to the fMRI data file.
    atlas_path : str
        Path to the brain atlas file.
    dataset_name : str
        Name of the dataset to display fMRI data from.
    hrf_delay : int
        fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
        that reflects changes in blood oxygenation levels in response to
        activity in the brain. Blood flow increases to a given brain region in
        response its activity. This vascular response, which follows the
        hemodynamic response function (HRF), takes time. Typically, the HRF
        peaks around 5–6 seconds after a neural event: this delay reflects the
        time needed for blood oxygenation changes to propagate and for the fMRI
        signal to capture them. Therefore, this parameter introduces a delay
        between stimulus chunks and fMRI samples. For example, with a hrf_delay
        of 3, if the stimulus chunk of interest is 17, the corresponding fMRI
        sample will be 20.

    """
    # Load the .tsv transcript data from the provided path
    transcript_df = load_transcript(transcript_path)

    # Create a dropdown widget with transcript text as options
    dropdown = create_dropdown_by_text(transcript_df)

    # Create an output widget to display transcript and brain visualization
    output = widgets.Output()

    # Define the function to handle dropdown value changes
    def on_chunk_select(change):
        with output:
            output.clear_output()  # Clear the previous output
            chunk_index = dropdown.value

            # Display transcript
            display_transcript(chunk_index, transcript_df)

            # Visualize brain fMRI data
            plot_fmri_on_brain(chunk_index, fmri_file_path, atlas_path,
                             dataset_name, hrf_delay)

    dropdown.observe(on_chunk_select, names='value')
    display(dropdown, output)
    
def list_splits(season):
	
    dir = os.path.join('data', 'friends_transcripts', season)
    splits_list = [
        x.split("/")[-1].split(".")[0][28:]
		for x in sorted(glob.glob(f"{dir}/*.tsv"))]
    
    return splits_list




def interactive_brain_slicer(mask_path, bg_template='mni152', cmap='autumn', 
                             alpha=0.7, title='Interactive Brain Slicer'):

    # Load the mask
    mask_img = nib.load(mask_path)
    
    # Load background template
    if bg_template == 'mni152':
        mni = datasets.load_mni152_template(resolution=2)
    else:
        mni = nib.load(bg_template)
    
    # Sliders
    x_slider = widgets.IntSlider(
        value=0, min=-90, max=90, step=2,
        description='X (sagittal):', continuous_update=False,
        style={'description_width': '120px'}
    )
    y_slider = widgets.IntSlider(
        value=-52, min=-126, max=90, step=2,
        description='Y (coronal):', continuous_update=False,
        style={'description_width': '120px'}
    )
    z_slider = widgets.IntSlider(
        value=10, min=-72, max=108, step=2,
        description='Z (axial):', continuous_update=False,
        style={'description_width': '120px'}
    )
    
    output = widgets.Output()
    
    def update_plot(change=None):
        with output:
            clear_output(wait=True)
            x, y, z = x_slider.value, y_slider.value, z_slider.value
            plotting.plot_roi(
                mask_img, bg_img=mni, cmap=cmap, alpha=alpha,
                title=f"{title} (x={x}, y={y}, z={z})",
                display_mode="ortho", cut_coords=(x, y, z),
                black_bg=False, colorbar=False
            )
            plotting.show()
    
    # collega gli slider
    x_slider.observe(update_plot, names='value')
    y_slider.observe(update_plot, names='value')
    z_slider.observe(update_plot, names='value')
    
    # layout e primo draw
    box = widgets.VBox([x_slider, y_slider, z_slider, output])
    ipy_display(box)
    update_plot()   
