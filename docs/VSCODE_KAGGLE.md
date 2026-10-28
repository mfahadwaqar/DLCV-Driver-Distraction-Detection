# Run a local VS Code notebook on Kaggle

Kaggle offers a **VSCode Compatible URL** for its interactive Jupyter Server. VS Code edits the local notebook while cells execute in the Kaggle session. The connection is for Jupyter notebook cells; it does not mount the local repository as a Kaggle filesystem or turn VS Code's normal terminal into a Kaggle terminal.

1. In the existing Kaggle notebook tab, set **Settings > Accelerator** to a GPU and start the interactive session. Open **Run > Kaggle Jupyter Server**. Wait for "Session must be running" to disappear, then copy the **VSCode Compatible URL**. Treat the URL as a private access token; do not put it in Git or share it.
2. Open [`notebooks/rt_bene_eye_state_training_kaggle.ipynb`](../notebooks/rt_bene_eye_state_training_kaggle.ipynb) from this repository in desktop VS Code. The Jupyter extension is already installed on this computer.
3. Select **Kernel** (top right) > **Select Another Kernel** > **Existing Jupyter Server** > **Enter the URL of the running Jupyter server**. Paste Kaggle's URL, then choose its Python kernel. If VS Code offers to trust the remote server, verify it is the Kaggle URL copied in step 1.
4. Run numbered Steps 1–3 in the notebook to check the connection, fetch the pinned project code, and prepare RT-BENE. The large dataset remains on Kaggle in `/kaggle/temp`. Step 3 reuses a completed manifest in the same session.
5. Step 4 is optional interactive training and defaults to off. If you turn it on, choose a new `RUN_NAME`; results go under `/kaggle/working/artifacts`. Step 5 shows only those interactive results. Download important files before the session ends. For a durable full run, use the saved-version CLI workflow in [KAGGLE.md](KAGGLE.md).

The notebook fetches a pinned GitHub commit on the Kaggle runtime. Edit code locally, commit and push it, then update `SOURCE_COMMIT` in the notebook to run that revision. A VS Code Jupyter connection does not automatically synchronize local source files. Kaggle's server URL can change when the interactive session ends, so reconnect with a fresh URL next time.

The generated saved Kaggle notebook and this readable local VS Code notebook are independent. Running local cells does not update the saved run. The local file was renamed for clarity; the earlier preparation output was backed up locally under the ignored `results/notebook-backups/` folder.
