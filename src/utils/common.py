import warnings


def ignore_warnings(categories=None):
    """
    여러 경고 카테고리를 무시하기

    - Warning: 모든 경고의 기본 클래스입니다.
    - DeprecationWarning: 특정 기능이 폐기 예정임을 알리는 경고입니다.
                          향후 버전에서 제거될 수 있음을 의미합니다.
    - PendingDeprecationWarning: 아직 폐기가 예정되지 않았지만, 미래에 폐기될 가능성이 있음을 알리는 경고입니다.
    - SyntaxWarning: 구문과 관련된 문제가 있을 때 발행되는 경고입니다.
    - RuntimeWarning: 실행 중 발생하는 경고로, 일반적인 런타임 경고를 나타냅니다.
    - UserWarning: 사용자가 직접 생성한 경고입니다.
    - FutureWarning: 현재는 사용 가능하지만, 향후에 변경될 가능성이 있는 기능에 대해 발행되는 경고입니다.
    - ImportWarning: 모듈 임포트와 관련된 문제에 대해 발행되는 경고입니다.
    - UnicodeWarning: 유니코드 관련 문제에 대해 발행되는 경고입니다.
    - BytesWarning: 바이트 코드 관련 문제에 대해 발행되는 경고입니다.
    - ResourceWarning: 리소스 사용과 관련된 경고입니다, 예를 들어 파일이나 소켓이 닫히지 않았을 때 발생합니다.

    :param categories: 무시할 경고 카테고리의 리스트입니다.

    :Usage:
        ignore_warnings()

        또는,
        # 무시할 경고 카테고리 리스트
        categories_to_ignore = [
            DeprecationWarning,
            PendingDeprecationWarning,
            FutureWarning
        ]
        ignore_warnings(categories=categories_to_ignore)
    """
    if categories is None:
        categories = [DeprecationWarning, PendingDeprecationWarning, FutureWarning]

    for category in categories:
        warnings.filterwarnings("ignore", category=category)
