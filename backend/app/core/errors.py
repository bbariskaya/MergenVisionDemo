class MergenVisionError(Exception):
    pass


class NotFoundError(MergenVisionError):
    pass


class ValidationError(MergenVisionError):
    pass


class ConflictError(MergenVisionError):
    pass


class InternalError(MergenVisionError):
    pass
