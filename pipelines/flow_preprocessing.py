from prefect import flow, task

from scripts.preprocessing.preprocessing import (
    create_and_save_train_val_data,
    load_all_events,
    preprocess_contents,
    preprocess_users,
)


@task
def task_preprocess_users():
    """Prefect task for user feature preprocessing."""
    return preprocess_users()


@task
def task_preprocess_contents():
    """Prefect task for content feature preprocessing."""
    return preprocess_contents()


@task
def task_load_events():
    """Prefect task for loading event data."""
    return load_all_events()


@task
def task_create_and_save_train_val_data(processed_users, train_X, val_X, events):
    create_and_save_train_val_data(processed_users, train_X, val_X, events)


@flow(name="Preprocessing Pipeline Flow")
def preprocessing_flow():
    """Main Prefect flow to run preprocessing pipeline."""
    print("🚀 Starting Prefect preprocessing pipeline...")

    # Step 1: Preprocess users
    processed_users = task_preprocess_users()

    # Step 2: Preprocess contents (returns train_X, val_X)
    train_X, val_X = task_preprocess_contents()

    # Step 3: Load events
    events = task_load_events()

    # Step 4: Create and save training data
    task_create_and_save_train_val_data(processed_users, train_X, val_X, events)

    print("🎉 Prefect preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    preprocessing_flow()
