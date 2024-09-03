from IPython.core.magic import register_cell_magic
from IPython import get_ipython

@register_cell_magic
def savefile(line, cell):
    """Save the content of the cell into a specified file and execute it."""
    filename = line.strip()
    if filename:
        with open(filename, "w") as file:
            file.write(cell)
        print(f"Cell content saved to {filename}")
    else:
        print("Error: Filename not specified")
    
    # Attempt to execute the cell content
    try:
        # This will execute the cell in the current IPython context and handle any output
        get_ipython().run_cell(cell)
    except Exception as e:
        print(f"An error occurred while executing the cell: {e}")

def load_ipython_extension(ipython):
    # This function is called when the extension is loaded.
    # It ensures that your magic is properly registered and available in IPython.
    pass
