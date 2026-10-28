# Run a local VS Code notebook on Kaggle

Kaggle offers a **VSCode Compatible URL** for its interactive Jupyter Server. VS Code edits the local notebook while cells execute in the Kaggle session. The connection is for Jupyter notebook cells; it does not mount the local repository as a Kaggle filesystem or turn VS Code's normal terminal into a Kaggle terminal.

1. In the existing Kaggle notebook tab, set **Settings > Accelerator** to a GPU and start the interactive session. Open **Run > Kaggle Jupyter Server**. Wait for "Session must be running" to disappear, then copy the **VSCode Compatible URL**. Treat the URL as a private access token; do not put it in Git or share it.
2. Open [`notebooks/kaggle_rt_bene.ipynb`](../notebooks/kaggle_rt_bene.ipynb) from this repository in desktop VS Code. The Jupyter extension is already installed on this computer.
3. Select **Kernel** (top right) > **Select Another Kernel** > **Existing Jupyter Server** > **Enter the URL of the running Jupyter server**. Paste Kaggle's URL, then choose its Python kernel. If VS Code offers to trust the remote server, verify it is the Kaggle URL copied in step 1.
4. Run the first cell. It asserts that execution is in `/kaggle/working` and that a CUDA GPU is available. Then run the code-fetch and RT-BENE preparation cells. The large dataset remains on Kaggle in `/kaggle/temp`.
5. For an interactive training experiment, set `RUN_TRAINING = True` in the training cell. Results go to a new folder under `/kaggle/working/artifacts`. Use a new `RUN_NAME` for every run. Download important files before the interactive session ends. For a durable full run, use the saved-version CLI workflow in [KAGGLE.md](KAGGLE.md).

The notebook fetches a pinned GitHub commit on the Kaggle runtime. Edit code locally, commit and push it, then update `SOURCE_COMMIT` in the notebook to run that revision. A VS Code Jupyter connection does not automatically synchronize local source files. Kaggle's server URL can change when the interactive session ends, so reconnect with a fresh URL next time.

The earlier CLI submitted a one-cell generated notebook as saved version 3. That version is visible from the notebook's version selector. The draft editor can still show the user's original single cell because it is separate from a saved run. Use this readable local notebook for interactive work; the saved Kaggle version remains the record of the initial smoke result.
