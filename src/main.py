import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import torch
import json
import numpy as np
from PIL import Image
import io
import shutil
import os
import zipfile
from io import BytesIO
from scripts.models import FineTunedEfficientNet
from scripts.helpers import test_transform
from omnixai.data.image import Image as omniImage
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM
from omnixai.explanations.image.pixel_importance import _plot_pixel_importance_heatmap

settings = get_settings()


class MyService(Service):
    """
    GradCAM XAI Service
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="GradCAM XAI",
            slug="gradcam-xai",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_ZIP]
                    # name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                    # name="result", type=[FieldDescriptionType.IMAGE_PNG]
                    # name="result", type=[FieldDescriptionType.IMAGE_JPEG]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.EXPLAINABlE_AI,
                    acronym=ExecutionUnitTagAcronym.EXPLAINABlE_AI,
                ),
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/gradcam-xai/",
        )
        self._logger = get_logger(settings)
        self._zip_path = "./gradcam_xai"

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type
        # before using them.

        raw = data["image"].data
        input_type = data["image"].type

        # output_type = data["format"].data.decode("utf-8")
        #
        stream = io.BytesIO(raw)
        img = omniImage(Image.open(stream).convert('RGB'))

        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get labels
        with open('data/idx_to_names.json', 'r') as file:
            class_dict = json.load(file)
        class_labels = list(class_dict.values())

        # Load model
        model = FineTunedEfficientNet()

        checkpoint = torch.load("EfficientNet_SportsImageClassification.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # Explain image
        def preprocess(ims):
            return torch.stack([test_transform()(im.to_pil()) for im in ims])

        gradcam = GradCAM(
            model=model,
            target_layer=model.model.features[-1][0],
            preprocess_function=preprocess
        )

        # Explain the top label
        explanations = gradcam.explain(img)
        index = 0
        exp = explanations.get_explanations(index)
        image_ndarray = exp["image"]
        scores_ndarray = exp["scores"]
        target_label = exp["target_label"]

        # transformation
        image = np.transpose(np.stack([image_ndarray] * 3), (1, 2, 0)) if image_ndarray.ndim == 2 else image_ndarray
        importance_scores = np.expand_dims(scores_ndarray, axis=-1) if scores_ndarray.ndim == 2 else scores_ndarray

        # Image and importance scores
        scores = _plot_pixel_importance_heatmap(importance_scores, image, overlay=True)

        # Convert the ndarray to a PIL Image
        image = Image.fromarray(scores)

        # Convert the PIL Image to bytes
        out_buff = io.BytesIO()
        image.save(out_buff, format=input_type.split('/')[1])
        image_heatmap = out_buff.getvalue()

        # json label output
        predicted_label = json.dumps({
            "predicted_label": class_labels[target_label]
        })

        # save files
        if os.path.exists(self._zip_path):
            shutil.rmtree(self._zip_path)

        if not os.path.exists(self._zip_path):
            os.makedirs(self._zip_path)

        label_file_path = os.path.join(self._zip_path, "predicted_label.json")
        with open(label_file_path, "w") as file:
            file.write(predicted_label)

        heatmap_file_path = os.path.join(self._zip_path, "heatmap.jpg")
        with open(heatmap_file_path, "wb") as file:
            file.write(image_heatmap)

        # zip files
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self._zip_path):
                for file in files:
                    zipf.write(
                        os.path.join(root, file),
                        os.path.relpath(
                            os.path.join(root, file),
                            os.path.join(self._zip_path, ".."),
                        ),
                    )

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(
                # data=predicted_label,
                # type=FieldDescriptionType.APPLICATION_JSON
                # data=image_heatmap,
                # type=input_type
                data=zip_buffer.getvalue(),
                type=FieldDescriptionType.APPLICATION_ZIP
            )
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """GradCAM XAI
Give a visual explanation of sport classification using GradCAM.
"""
api_summary = """GradCAM XAI service
Give a visual explanation for classification decisions of sport images using GradCAM.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="GradCAM XAI Service API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
