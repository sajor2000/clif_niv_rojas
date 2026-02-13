import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NIPPV Wide Dataset Generation

    This notebook creates a wide dataset of ICU patients with NIPPV events for analysis.

    ## Objective
    Generate `study_cohort_NIPPV_&_ICU.csv` containing:
    - All ICU patients with at least one NIPPV event
    - Event-level data (vitals, labs, respiratory support, assessments)
    - Age ≥ 18 years
    - Ready for filtering and analysis in downstream notebooks

    ## Process
    1. Pre-filter: Identify hospitalizations with NIPPV events
    2. Load all tables filtered to those hospitalizations
    3. Apply age filter
    4. Apply outlier handling
    5. Create wide dataset
    6. Save as CSV
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup and Configuration
    """)
    return


@app.cell
def _():
    import os
    import pandas as pd
    from clifpy.clif_orchestrator import ClifOrchestrator
    from clifpy.utils.outlier_handler import apply_outlier_handling
    import json
    import warnings
    warnings.filterwarnings('ignore')

    print("=== NIPPV Wide Dataset Generation ===" )
    print("Setting up environment...")
    return ClifOrchestrator, apply_outlier_handling, json, os, pd


@app.cell
def _(json, os):
    # Load configuration
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    project_root = os.path.dirname(script_dir)  # Parent of code/ directory

    # Try to find config in project root first, then current directory
    config_path = None
    for path in [
        os.path.join(project_root, 'config.json'),
        os.path.join(project_root, 'clif_config.json'),
        'config.json',
        'clif_config.json'
    ]:
        if os.path.exists(path):
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError("Could not find config.json or clif_config.json")

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print(f"Site: {config['site']}")
    print(f"Data path: {config['data_directory']}")
    print(f"File type: {config['filetype']}")

    # Output directory in project root
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return config_path, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pre-Filter: Identify NIPPV Hospitalizations
    """)
    return


@app.cell
def _(ClifOrchestrator, config_path):
    # Initialize ClifOrchestrator
    print("Initializing ClifOrchestrator for pre-filtering...")
    clif_prefilter = ClifOrchestrator(config_path=config_path)

    # Load ONLY respiratory_support table to identify NIPPV patients
    print("\nLoading respiratory_support table to identify NIPPV hospitalizations...")
    clif_prefilter.load_table('respiratory_support')

    # Get hospitalization IDs with NIPPV
    resp_df = clif_prefilter.respiratory_support.df
    nippv_hosp_ids = resp_df[resp_df['device_category'] == 'NIPPV']['hospitalization_id'].unique().tolist()

    print(f"\n[OK] Found {len(nippv_hosp_ids):,} hospitalizations with NIPPV events")
    print(f"This will be used to filter all subsequent table loads (reduces memory usage)")
    return (nippv_hosp_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Feature Categories
    """)
    return


@app.cell
def _():
    # Define category filters for NIPPV cohort features
    print("Configuring feature categories...")

    category_filters = {
        'vitals': [
            'heart_rate', 'map', 'sbp', 'respiratory_rate', 'spo2', 'temp_c',
            'weight_kg', 'height_cm'
        ],
        'labs': [
            'pco2_arterial', 'ph_arterial', 'pco2_venous', 'ph_venous'
        ],
        'respiratory_support': [
            'device_category', 'fio2_set', 'peep_set',
            'peak_inspiratory_pressure_obs', 'tidal_volume_obs'
        ],
        'patient_assessments': [
            'gcs_total'
        ]
    }

    print("\nFeature categories:")
    for table, categories in category_filters.items():
        print(f"  {table}: {len(categories)} categories")
    return (category_filters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Tables with NIPPV Filter
    """)
    return


@app.cell
def _(ClifOrchestrator, category_filters, config_path, nippv_hosp_ids):
    # Re-initialize ClifOrchestrator for full data load
    print("\nRe-initializing ClifOrchestrator for full table loading...")
    clif = ClifOrchestrator(config_path=config_path)

    # Define table loading configuration
    table_config = {
        'vitals': {
            'columns': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
            'category_filter': ('vital_category', category_filters['vitals'])
        },
        'labs': {
            'columns': ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric'],
            'category_filter': ('lab_category', category_filters['labs'])
        },
        'respiratory_support': {
            'columns': None,
            'category_filter': None
        },
        'patient_assessments': {
            'columns': None,
            'category_filter': ('assessment_category', category_filters['patient_assessments'])
        },
        'hospitalization': {
            'columns': None,
            'category_filter': None
        },
        'patient': {
            'columns': None,
            'category_filter': None
        }
    }

    # Load tables with NIPPV hospitalization ID filter
    print(f"\nLoading tables filtered to {len(nippv_hosp_ids):,} NIPPV hospitalizations...")
    for tbl_name, tbl_config in table_config.items():
        # Build filters dict - patient table doesn't have hospitalization_id column
        if tbl_name == 'patient':
            filters_dict = None  # Patient table: load without filter (small table, joins via patient_id)
            print(f"Loading {tbl_name}: all patients (no filter, joins via patient_id)...")
        else:
            filters_dict = {'hospitalization_id': nippv_hosp_ids}

            # Add category filter if specified
            if tbl_config.get('category_filter'):
                category_col, category_values = tbl_config['category_filter']
                filters_dict[category_col] = category_values
                print(f"Loading {tbl_name}: {len(nippv_hosp_ids):,} hosps × {len(category_values)} categories...")
            else:
                print(f"Loading {tbl_name}: {len(nippv_hosp_ids):,} hospitalizations...")

        clif.load_table(
            tbl_name,
            filters=filters_dict,
            columns=tbl_config.get('columns')
        )

    print("\n[OK] All tables loaded successfully with NIPPV filter")
    return (clif,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Calculate and Add BMI to Patient Assessments
    """)
    return


@app.cell
def _(clif, pd):
    # Calculate BMI from vitals table (will be added to wide dataset later)
    print("Calculating BMI from vitals table (weight_kg and height_cm)...")

    vitals_df = clif.vitals.df.copy()

    # Extract weight and height from vitals
    weight_df = vitals_df[vitals_df['vital_category'] == 'weight_kg'][['hospitalization_id', 'vital_value']].copy()
    weight_df['weight_kg'] = pd.to_numeric(weight_df['vital_value'], errors='coerce')

    height_df = vitals_df[vitals_df['vital_category'] == 'height_cm'][['hospitalization_id', 'vital_value']].copy()
    height_df['height_cm'] = pd.to_numeric(height_df['vital_value'], errors='coerce')

    # Get first non-null values per hospitalization
    weight_first = weight_df.groupby('hospitalization_id')['weight_kg'].first().reset_index()
    height_first = height_df.groupby('hospitalization_id')['height_cm'].first().reset_index()

    # Merge and calculate BMI
    bmi_calc = pd.merge(weight_first, height_first, on='hospitalization_id', how='outer')
    bmi_calc['bmi'] = bmi_calc['weight_kg'] / ((bmi_calc['height_cm'] / 100) ** 2)

    # Keep only hospitalization_id and bmi for later merge
    bmi_for_merge = bmi_calc[['hospitalization_id', 'bmi']].copy()

    print(f"[OK] BMI calculated (will be merged to wide dataset later)")
    print(f"Hospitalizations with BMI: {bmi_calc['bmi'].notna().sum():,}")
    if bmi_calc['bmi'].notna().sum() > 0:
        print(f"BMI range: {bmi_calc['bmi'].min():.1f} - {bmi_calc['bmi'].max():.1f}")
        print(f"BMI median: {bmi_calc['bmi'].median():.1f}")
    return (bmi_for_merge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Apply Age Filter
    """)
    return


@app.cell
def _(clif, pd):
    # Apply age filter to hospitalization table
    print("Applying age filter (≥18 years)...")

    hosp_df = clif.hospitalization.df.copy()
    hosp_df['age_at_admission'] = pd.to_numeric(hosp_df['age_at_admission'], errors='coerce')

    initial_count = len(hosp_df)
    hosp_df = hosp_df[
        (hosp_df['age_at_admission'] >= 18) &
        (hosp_df['age_at_admission'].notna())
    ]
    filtered_count = len(hosp_df)

    print(f"Hospitalizations before age filter: {initial_count:,}")
    print(f"Hospitalizations after age filter (≥18): {filtered_count:,}")
    print(f"Removed: {initial_count - filtered_count:,}")

    # Update clif object
    clif.hospitalization.df = hosp_df

    # Get age-filtered IDs
    age_filtered_ids = hosp_df['hospitalization_id'].astype(str).unique().tolist()
    print(f"\n[OK] Age filter applied: {len(age_filtered_ids):,} unique hospitalizations")

    # Filter other tables to match age-filtered IDs
    print("\nFiltering other tables to match age-filtered hospitalizations...")
    for table_name in ['vitals', 'labs', 'respiratory_support', 'patient_assessments']:
        table_obj = getattr(clif, table_name)
        if table_obj is not None and hasattr(table_obj, 'df') and table_obj.df is not None:
            initial_rows = len(table_obj.df)
            table_obj.df = table_obj.df[table_obj.df['hospitalization_id'].isin(age_filtered_ids)]
            print(f"  {table_name}: {initial_rows:,} → {len(table_obj.df):,} rows")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Apply Outlier Handling
    """)
    return


@app.cell
def _(apply_outlier_handling, clif):
    # Apply outlier handling to tables using clifpy
    print("Applying outlier handling to loaded tables...")

    tables_for_outlier_handling = ['vitals', 'labs', 'respiratory_support', 'patient_assessments']

    for outlier_table_name in tables_for_outlier_handling:
        outlier_table_obj = getattr(clif, outlier_table_name)
        if outlier_table_obj is not None and hasattr(outlier_table_obj, 'df') and outlier_table_obj.df is not None:
            print(f"\nProcessing {outlier_table_name} table:")
            apply_outlier_handling(outlier_table_obj)
        else:
            print(f"Warning: {outlier_table_name} table not loaded or empty")

    print("\n[OK] Outlier handling completed")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Wide Dataset
    """)
    return


@app.cell
def _(category_filters, clif):
    # Create wide dataset
    print("Creating wide dataset using ClifOrchestrator...")
    print("Note: Loading ALL event data (no time filtering)")

    clif.create_wide_dataset(
        category_filters=category_filters,
        cohort_df=None,  # No time filtering
        save_to_data_location=False,
        batch_size=10000,
        memory_limit='6GB',
        threads=4,
        show_progress=True
    )

    wide_df = clif.wide_df.copy()

    print(f"\n[OK] Wide dataset created")
    print(f"Shape: {wide_df.shape}")
    print(f"Hospitalizations: {wide_df['hospitalization_id'].nunique():,}")
    print(f"Date range: {wide_df['event_time'].min()} to {wide_df['event_time'].max()}")
    return (wide_df,)


@app.cell
def _(bmi_for_merge, pd, wide_df):
    # Merge BMI as a column to wide dataset
    print("\nMerging BMI to wide dataset...")

    wide_df_with_bmi = pd.merge(
        wide_df,
        bmi_for_merge,
        on='hospitalization_id',
        how='left'
    )

    print(f"[OK] BMI merged to wide dataset")
    print(f"Shape: {wide_df_with_bmi.shape}")
    print(f"BMI column added - Non-null BMI values: {wide_df_with_bmi['bmi'].notna().sum():,}")
    return (wide_df_with_bmi,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Add Hospitalization Metadata
    """)
    return


@app.cell
def _(clif, pd, wide_df_with_bmi):
    # Merge hospitalization metadata and patient demographics
    print("\nAdding hospitalization metadata and patient demographics...")
    print("Note: age_at_admission and patient_id already in wide_df from create_wide_dataset()")
    print("Note: bmi added in previous step via left join")

    # Step 1: Get hospitalization metadata (EXCLUDE age_at_admission - already in wide_df!)
    hosp_metadata = clif.hospitalization.df[
        ['hospitalization_id', 'patient_id', 'admission_dttm', 'discharge_dttm', 'discharge_category']
    ].copy()
    hosp_metadata['hospitalization_id'] = hosp_metadata['hospitalization_id'].astype(str)
    hosp_metadata['admission_dttm'] = pd.to_datetime(hosp_metadata['admission_dttm'])
    hosp_metadata['discharge_dttm'] = pd.to_datetime(hosp_metadata['discharge_dttm'])

    # Step 2: Get patient demographics (sex_category needed by Rush notebook)
    patient_demo = clif.patient.df[['patient_id', 'sex_category']].copy()

    # Step 3: Merge patient demographics into hospitalization metadata
    hosp_with_demo = pd.merge(
        hosp_metadata,
        patient_demo,
        on='patient_id',
        how='left'
    )

    # Step 4: Merge into wide dataset and clean up
    wide_df_with_metadata = pd.merge(
        wide_df_with_bmi,
        hosp_with_demo.drop(columns=['patient_id']),
        on='hospitalization_id',
        how='left'
    ).drop(columns=['patient_id'], errors='ignore')  # Remove patient_id from final output

    print(f"[OK] Metadata and demographics merged")
    print(f"Added columns: admission_dttm, discharge_category, sex_category")
    print(f"Retained columns: age_at_admission, bmi")
    print(f"Shape: {wide_df_with_metadata.shape}")
    return (wide_df_with_metadata,)


@app.cell
def _(wide_df_with_metadata):
    # Display sample
    wide_df_with_metadata.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save Wide Dataset
    """)
    return


@app.cell
def _(os, output_dir, wide_df_with_metadata):
    # Save wide dataset as CSV (matching Rush notebook input format)
    print("\n" + "=" * 60)
    print("SAVING WIDE DATASET")
    print("=" * 60)

    output_csv_path = os.path.join(output_dir, 'study_cohort_NIPPV_&_ICU.csv')
    wide_df_with_metadata.to_csv(output_csv_path, index=False)

    print(f"\n[OK] Wide dataset saved to: {output_csv_path}")
    print(f"File size: {os.path.getsize(output_csv_path) / 1024**2:.1f} MB")
    print(f"Shape: {wide_df_with_metadata.shape}")
    print(f"Columns: {len(wide_df_with_metadata.columns)}")

    # Also save as parquet for efficiency
    output_parquet_path = os.path.join(output_dir, 'study_cohort_NIPPV_&_ICU.parquet')
    wide_df_with_metadata.to_parquet(output_parquet_path, index=False)
    print(f"\n[OK] Parquet version saved to: {output_parquet_path}")
    print(f"File size: {os.path.getsize(output_parquet_path) / 1024**2:.1f} MB")

    print("\n" + "=" * 60)
    print("[SUCCESS] Wide Dataset Generation Completed!")
    print("=" * 60)
    print(f"\nThis dataset can now be used as input to:")
    print(f"  - Rush NIPPV Study Cohort.ipynb")
    print(f"  - Any other NIPPV cohort analysis notebook") 
    return


if __name__ == "__main__":
    app.run()
