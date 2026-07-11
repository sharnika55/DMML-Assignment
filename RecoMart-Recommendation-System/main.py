from orchestration.prefect_pipeline import run_prefect_pipeline


def main() -> None:
    print("RecoMart Recommendation System")
    run_prefect_pipeline()


if __name__ == "__main__":
    main()
