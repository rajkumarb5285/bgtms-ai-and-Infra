# M1-04: Dataset Research & Analysis — BGTMS

## 1. Overview of Evaluated Datasets
To accurately capture the chaotic, mixed, and unlane-disciplined traffic of Bengaluru, three key datasets were selected and hosted in the Amazon S3 bucket (`s3://bgtms-ai-datasets/`):

| Dataset | Size in S3 | Focus Area | Strengths for Bengaluru Traffic | Limitations |
|---|---|---|---|---|
| **Indian Driving Dataset (IDD)** | 5.5 GiB | Real Indian driving conditions (Hyderabad & Bengaluru roads) | - Unstructured roadways<br>- High density of 2-wheelers & autos<br>- Dense pedestrian and roadside activity | Heavy segmentation masks; needs bounding box extraction. |
| **Indian Vehicle Dataset** | 330.0 MiB | Dedicated Indian vehicle object classes | - Auto-rickshaws, bikes, cars, buses, trucks<br>- Clean bounding box annotations<br>- High camera diversity | Smaller scale than full IDD. |
| **UA-DETRAC** | 9.2 GiB | Dense traffic intersection tracking & vehicle detection | - Multitude of weather/lighting conditions<br>- Detailed occlusion & truncation attributes<br>- Benchmark tracking sequences | Western/Chinese vehicle types (lacks auto-rickshaws). |

## 2. Harmonized Class Taxonomy for BGTMS
To unify disparate datasets into a production detector for Bengaluru traffic, the following standard 8-class taxonomy is established:

| Class ID | Class Name | Description | IRC PCU Equivalent |
|---|---|---|---|
| 0 | `car` | Sedans, hatchbacks, SUVs, taxis | 1.00 |
| 1 | `motorcycle` | Motorcycles, scooters, mopeds | 0.50 |
| 2 | `auto_rickshaw` | 3-wheeled passenger/cargo autos | 1.00 |
| 3 | `bus` | BMTC buses, private coaches | 3.00 |
| 4 | `truck` | Light/heavy commercial vehicles, tempos | 3.00 |
| 5 | `bicycle` | Non-motorized pedal cycles | 0.40 |
| 6 | `pedestrian` | People crossing or walking on carriageway | 0.20 |
| 7 | `emergency_vehicle`| Ambulances, fire tenders, police interceptors | Priority / 0.00 |

## 3. Storage Strategy & S3 Sync Plan
Given the available local disk capacity (~18.9 GB), downloading all 15.1 GiB zip archives simultaneously could trigger disk exhaustion during unzipping.
- **Phase 1 (Immediate Extraction):** Download and extract `raw/indian-vehicle/indian-vehicle-dataset.zip` (330 MB).
- **Phase 2 (Selective Sampling):** Extract targeted subsets and verification sequences from IDD and UA-DETRAC.
- **Phase 3 (Production S3 Integration):** Cloud-native data loader capable of streaming directly from S3 during high-scale training.
