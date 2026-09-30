# Practice YOLO product dataset

56 upright JPEGs, 56 matching labels, 12 classes. Train: 44; validation: 12.
The second image in filename order for each product is assigned to validation.

Images are copied unchanged from computer_vision/data/catalog_annotations/images.
Draft main-product bounding boxes are reused from the neighboring labels folder;
class IDs are remapped to preserve the earlier five-class scheme. data.yaml is authoritative.
No model was trained. Original source files are untouched.
Raw HEIC originals are copied into raw/catalog_originals. Upright JPEG inputs are
copied into raw/upright_jpegs. Unique legacy dataset photos are preserved in
raw/legacy_dataset. The prior dataset folder and training script are archived
under backups/dataset_consolidation_20260917 at the project root.
The training entry point now uses data/data.yaml; it has not been run.

Cautions: boxes are manually estimated drafts requiring review. Only the main product
is labeled; incidental background products are NOT exhaustively annotated. These labels
were initially prepared for catalog crops, not complete whole-image detector supervision.
Before serious detector training, annotate background products or remove/mask those regions.
All photos are from one capture session; validation is a workflow check, not independent
performance evidence. Collect separate sessions, backgrounds and lighting for evaluation.

Structure: images/train, images/val, labels/train, labels/val, data.yaml.
Review original box overlays in computer_vision/data/catalog_annotations/review.html.
