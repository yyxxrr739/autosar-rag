docker run -itd --name=qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -e "QDRANT__SERVICE__API_KEY=123456" \
    -e "QDRANT__SERVICE__JWT_RBAC=true" \
    -v /home/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant
