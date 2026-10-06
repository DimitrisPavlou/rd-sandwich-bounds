"""Data sources and the DataLoader factory used by the trainers.

``datasets.get_dataset`` turns a ``--dataset`` spec into a dataset;
``base.build_loader`` turns any dataset into a DataLoader of ``x`` batches. Synthetic sources
(Gaussian, banana) sample on demand; finite ones (arrays, image folders) are
map-style datasets that also expose ``.sample(batchsize)``.
"""
