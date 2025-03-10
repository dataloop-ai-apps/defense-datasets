import json
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import dtlpy as dl
import requests

logger = logging.getLogger(name='Military Assets Dataset')


class MilitaryAssetsDataset(dl.BaseServiceRunner):
    def __init__(self):
        self.logger = logger
        self.logger.info('Initializing dataset loader')

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress=None):
        """
        Uploads a text dataset to the specified destination.

        Args:
            dataset (dl.Dataset): The dataset object to upload.
            source (str): The source URL of the dataset.
            progress (optional): An optional progress object to track the upload progress.

        Returns:
            None

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.
            zipfile.BadZipFile: If the downloaded zip file is corrupted.
            KeyError: If there is an issue with the dataset metadata or configuration.
        """

        if progress is not None:
            progress.update(
                progress=0,
                message='Creating dataset...',
                status='Creating dataset...'
            )

        self.logger.info('Downloading zip file...')
        direc = os.getcwd()
        zip_dir = os.path.join(direc, "military-dataset.zip")

        if not os.path.exists(zip_dir):
            response = requests.get(source, timeout=100)
            if response.status_code == 200:
                with open(zip_dir, 'wb') as f:
                    f.write(response.content)
            else:
                self.logger.error(
                    'Failed to download the file. Status code: %s', response.status_code
                )
                return

        with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
            zip_ref.extractall(direc)
        self.logger.info('Zip file downloaded and extracted.')

        if progress is not None:
            progress.update(
                progress=0,
                message='Uploading items and annotations ...',
                status='Uploading items and annotations ...',
            )

        progress_tracker = {'last_progress': 0}

        def progress_callback_all(progress_class, progress, context):
            new_progress = progress // 2
            if (
                new_progress > progress_tracker['last_progress']
                and new_progress % 5 == 0
            ):
                logger.info(f'Progress: {new_progress}%')
                progress_tracker['last_progress'] = new_progress
                if progress_class is not None:
                    progress_class.update(
                        progress=new_progress,
                        message='Uploading items and annotations ...',
                        status='Uploading items and annotations ...',
                    )

        progress_callback = partial(progress_callback_all, progress)

        dl.client_api.add_callback(
            func=progress_callback, event=dl.CallbackEvent.ITEMS_UPLOAD
        )

        # Upload features
        vectors_file = os.path.join(direc, 'features.json')
        with open(vectors_file, 'r') as f:
            vectors = json.load(f)

        annotations_files = os.path.join(direc, 'json/')
        items_files = os.path.join(direc, 'items/')
        dataset.items.upload(
            local_path=items_files,
            local_annotations_path=annotations_files,
        )

        # Setup dataset recipe and ontology
        recipe = dataset.recipes.list()[0]
        ontology = recipe.ontologies.list()[0]
        with open(os.path.join(direc, 'Military Assets-ontology.json'), 'r') as f:
            ontology_json = json.load(f)
        ontology.copy_from(ontology_json=ontology_json)

        feature_set = self.ensure_feature_set(dataset)

        with ThreadPoolExecutor(max_workers=32) as executor:
            vector_features = [
                executor.submit(self.create_feature, dataset, item_json_data, feature_set)
                for item_json_data in vectors
            ]

            self.upload_progress(
                progress=progress,
                futures=vector_features,
                message='Uploading feature set ...',
                min_progress=50,
                max_progress=100
            )

        self.logger.info('Dataset uploaded successfully')

    @staticmethod
    def upload_progress(progress, futures, message, min_progress, max_progress):
        """
        Tracks and logs the progress of a set of asynchronous tasks.

        Args:
            progress (object): An object that has an `update` method to report progress.
            futures (list): A list of futures representing the asynchronous tasks.
            message (str): A message to be logged and passed to the progress object.
            min_progress (int): The minimum progress value (usually 0).
            max_progress (int): The maximum progress value (usually 100).

        Logs:
            Logs the progress percentage at each step.

        Updates:
            Calls the `update` method of the `progress` object with the new progress value and message.
        """
        total_tasks = len(futures)
        tasks_completed = 0
        task_progress = 0
        for _ in as_completed(futures):
            tasks_completed += 1
            new_progress = (
                int(tasks_completed / total_tasks * (max_progress - min_progress))
                + min_progress
            )
            if new_progress > task_progress and new_progress % 1 == 0:
                logger.info(f'Progress: {new_progress}%')
                task_progress = new_progress
                if progress is not None:
                    progress.update(
                        progress=new_progress,
                        message=message,
                        status=message
                    )

    @staticmethod
    def ensure_feature_set(dataset: dl.Dataset):
        """
        Ensures that the feature set exists or creates a new one if not found.

        :param dataset: The dataset where the feature set is to be managed.
        """
        try:
            feature_set = dataset.project.feature_sets.get(
                feature_set_name='clip-feature-set'
            )
            logger.info(
                'Feature Set found! Name: %s, ID: %s', feature_set.name, feature_set.id
            )
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found, creating...')
            feature_set = dataset.project.feature_sets.create(
                name='clip-feature-set',
                entity_type=dl.FeatureEntityType.ITEM,
                project_id=dataset.project.id,
                set_type='clip',
                size=512,
            )
        return feature_set

    @staticmethod
    def create_feature(dataset: dl.Dataset, source_item_json_data, feature_set):
        """
        Creates a feature for a given item.

        :param dataset: The dataset containing the items.
        :param source_item_json_data: The source item data in json format (from the exported data).
        :param feature_set: The feature set to which the feature will be added.
        """

        source_item: dl.Item = dl.Item.from_json(_json=source_item_json_data, client_api=dl.client_api)
        target_item = dataset.items.get(filepath=source_item.filename)
        feature_set.features.create(entity=target_item, value=source_item_json_data["itemVectors"][0]["value"])
