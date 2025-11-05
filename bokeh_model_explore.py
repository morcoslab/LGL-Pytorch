import os
from itertools import cycle

import numpy as np
import pandas as pd
from Bio import SeqIO
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColorBar,
    ColorPicker,
    ColumnDataSource,
    Div,
    Dropdown,
    HoverTool,
    LassoSelectTool,
    LinearColorMapper,
    PreText,
    TabPanel,
    Tabs,
    Title,
)
from bokeh.palettes import (
    Cividis256,
    Colorblind8,
    Inferno256,
    Magma256,
    Set3,
    Turbo256,
    Viridis256,
    linear_palette,
)
import torch


### For now, comment out line 289 if you want to save as PNG instead of SVG
# bokeh
from bokeh.plotting import figure
from lglvae.lglvae import LGLVAE
from matplotlib.colors import to_hex, to_rgb
from skimage.draw import polygon

# directory setup
model_folder = os.curdir
msa_folder = os.curdir
data_folder = os.curdir


def get_subfolders(path):
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return sorted(subfolders)


def get_files(path):
    files = [f for f in os.listdir(path) if not os.path.isdir(os.path.join(path, f))]
    return sorted(files)


model_directory = get_files(model_folder)
msa_directory = get_files(msa_folder)
data_directory = get_files(data_folder)


# variables that must be predefined and then updated by widgets
class mutable_variables:
    def __init__(self):
        self.the_model = ""
        self.df = ""
        self.base_cds = ""
        self.grid_hamiltonian_plot = ""
        self.model_folder = model_folder
        self.msa_folder = msa_folder
        self.data_folder = data_folder
        self.grid_hamiltonian_df = ""
        self.base_plot = ""
        self.label_df = ""
        self.ldf_labels = list()
        self.unmodified_label_df = ""
        self.grid_ranges = []
        self.pixels = 0
        self.plot_is_gradient = True
        self.index_grid = ""
        self.landscape_seq_df = ""
        self.selected_positions = []
        self.grid_seq_location = ""
        self.bp_color_size = ["steelblue", 6]

    def update_model(self, model_name):
        self.the_model = LGLVAE.load(os.path.join(model_folder, model_name))

    def update_df(self, pandas_df):
        self.df = pandas_df

    def update_base(self, plotscatter):
        self.base_plot = plotscatter

    def update_grid_df(self, dataframe):
        self.grid_hamiltonian_df = dataframe

    def update_grid_plot(self, plot_return):
        self.grid_hamiltonian_plot = plot_return

    def update_labeldf(self, labeldf):
        self.label_df = labeldf

    def update_ldf_labels(self, labeldf):
        self.ldf_labels = list(labeldf.columns.values)

    def update_basecds(self, cds):
        self.base_cds = cds

    def update_cds_column(self, labeldf_column):
        self.base_cds.data[labeldf_column] = self.label_df[labeldf_column]

    def update_colors(self, colorlist):
        self.base_cds.data["colors"] = colorlist

    def update_labels(self, labellist):
        self.base_cds.data["Labels"] = labellist
        self.unmodified_label_df = pd.DataFrame(
            self.base_cds.data
        )  # happens every time you do CSV column selection

    def change_plot_from_selection(self, unselected_label_list):
        newdf = (
            self.unmodified_label_df
        )  # builds displayed selection from scratch, every time
        for label in unselected_label_list:
            newdf = newdf[newdf["Labels"] != label]
        self.base_cds.data.update(newdf)
        self.update_df(newdf)
        self.base_plot.data_source.selected.on_change("indices", select_points)

    def update_grid_ranges(self, extent_array):
        self.grid_ranges = extent_array


lm = mutable_variables()


# Color Functions
def gray_to_rgb_gradient_np(gray_value, target_rgb, n):
    start = np.array([gray_value, gray_value, gray_value])
    end = np.array(target_rgb)
    # Create an array of interpolation factors from 0 to 1 (length n)
    t = np.linspace(0, 1, n).reshape(n, 1)
    # Linearly interpolate between start and end for each channel
    gradient = (1 - t) * start + t * end
    return gradient


def gencolor(numpoints, selected_color=None):
    if not selected_color:
        colors = [np.random.randint(20, 150) / 255 for _ in range(3)]
    else:
        colors = to_rgb(selected_color)
    color_gradient = gray_to_rgb_gradient_np(105 / 255, colors, numpoints)
    return [to_hex(x) for x in color_gradient]


twofiftysix = cycle([Turbo256, Viridis256, Magma256, Cividis256, Inferno256])
solid_cycler = cycle(Colorblind8)
next(solid_cycler)

VAE_custom_legend_pallet = [
    "#ee6222",
    "#eeb422",
    "#898f9b",
    "#ee22c2",
    "#6e5410",
    "#b422ee",
    "#ffe755",
    "#c3941c",
    "#ee4e22",
    "#ee22d6",
    "#54106e",
    "#8b22ee",
    "#f59a81",
    "#ee22ae",
    "#b2b349",
    "#7c7d56",
    "#28281b",
    "#b7d290",
    "#d2909c",
    "#d290c4",
    "#5f49b3",
    "#fcadc4",
    "#ab90d2",
    "#b34967",
    "#c2a0aa",
    "#49b395",
    "#f5ae3d",
    "#f5643d",
    "#c27b0a",
    "#fad69e",
    "#fab19e",
    "#d1ffff",
    "#6bffff",
    "#bf4f4f",
    "#685665",
    "#8c321c",
] + list(Set3[12])
VAE_custom_legend_pallet = cycle(VAE_custom_legend_pallet)


# Bokeh functions
def select_the_model(event):
    lm.update_model(event.item)
    plot_grid_hamiltonian()
    t.text = event.item


def plot_grid_hamiltonian():  # plots selected grid_dataset.pkl
    print("pickle loaded")
    lm.bp_color_size = ["white", 3]
    p.x_range.range_padding = p.y_range.range_padding = 0
    p.grid.grid_line_width = 0.5
    grid_dataset = lm.the_model.LGL
    pixels = grid_dataset[grid_dataset[:, 0] == grid_dataset[0, 0]].shape[0]
    lm.pixels = pixels
    image_grid = np.zeros((pixels, pixels))
    # index_grid = np.zeros((pixels, pixels))
    count = 0
    for i in range(pixels):
        for j in range(pixels):
            image_grid[j, i] = grid_dataset[count][2]
            # index_grid[j, i] = count
            count += 1
    # lm.index_grid = np.flipud(np.rot90(index_grid))
    xmin, ymin = grid_dataset[0][0], grid_dataset[0][1]
    xmax, ymax = grid_dataset[-1][0], grid_dataset[-1][1]
    lm.update_grid_ranges([xmin, ymin, xmax, ymax])
    xspan, yspan = grid_dataset[-1][0] - xmin, grid_dataset[-1][1] - ymin

    color_mapper = LinearColorMapper(
        palette=Viridis256,
        low=grid_dataset[:, 2].min(),
        high=grid_dataset[:, 2].max(),
    )
    output_plot = p.image(
        image=[image_grid],
        x=xmin,
        y=ymin,
        dw=xspan,
        dh=yspan,
        level="image",
        color_mapper=color_mapper,
    )
    output_plot.glyph.color_mapper = color_mapper
    # Remove any existing color bars before adding a new one
    for layout in list(p.center) + list(p.right):
        if isinstance(layout, ColorBar):
            p.center.remove(layout) if layout in p.center else p.right.remove(layout)
    color_bar = ColorBar(color_mapper=color_mapper, width=5, label_standoff=5)
    color_bar.visible = True
    p.add_layout(color_bar, "right")
    lm.update_grid_plot(output_plot)


def plot_base_data(event):  # plots primary dataset, selectable by lasso
    print("fasta upload succeed")
    newlatent = lm.the_model.encode_sequences(os.path.join(msa_folder, event.item))
    # build CDS
    newheaders = [
        x.description
        for x in SeqIO.parse(os.path.join(msa_folder, event.item), "fasta")
    ]
    new_data_dictionary = {"Name": newheaders}
    new_data_dictionary["Sequences"] = [
        str(x.seq) for x in SeqIO.parse(os.path.join(msa_folder, event.item), "fasta")
    ]
    for dimension in range(newlatent.shape[1]):
        new_data_dictionary["z" + str(dimension)] = newlatent[:, dimension]
    new_data_dictionary["colors"] = [
        lm.bp_color_size[0] for _ in range(newlatent.shape[0])
    ]
    new_data_dictionary["Labels"] = ["Training Data" for _ in range(newlatent.shape[0])]
    lm.update_df(pd.DataFrame(data=new_data_dictionary))
    src = ColumnDataSource(lm.df)
    lm.update_basecds(src)
    # plot glyph
    base = p.scatter(
        "z0",
        "z1",
        fill_color="colors",
        line_color=None,
        size=lm.bp_color_size[1],
        legend_field="Labels",
        source=lm.base_cds,
        muted_alpha=0.2,
    )
    lm.update_base(base)
    add_colorpicker(base.glyph, title=event.item)

    lm.base_plot.data_source.selected.on_change("indices", select_points)
    p.legend.location = "top_left"
    p.legend.click_policy = "mute"
    if lm.bp_color_size[0] == "steelblue":
        p.x_range.range_padding = p.y_range.range_padding = 0.75


def load_grid_fasta(event):  # loads large grid msa as fasta.
    # make sure the grid_dataset.pkl and the grid_msa.fasta have equivalent lengths and are correctly indexed
    print("setting up parser...")
    lm.grid_seq_location = "hamiltonian_map/" + event.item


def add_colorpicker(glyph, title: str) -> None:
    picker = ColorPicker(title=title)
    # picker.js_link("color", glyph, "line_color")
    picker.js_link("color", glyph, "fill_color")
    main_layout.children[-2].children.append(picker)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")  # Remove '#' if present
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def add_colorpicker_wgradient(glyph, callback_fn, title: str) -> None:
    picker = ColorPicker(title=title)
    picker.on_change("color", callback_fn)
    main_layout.children[-2].children.append(picker)


def plot_data(event):  # plots any additional datapoints, unselectable by lasso
    print("fasta upload succeed")
    newlatent = lm.the_model.encode_sequences(os.path.join(msa_folder, event.item))
    newheaders = [
        x.description
        for x in SeqIO.parse(os.path.join(msa_folder, event.item), "fasta")
    ]
    new_data_dictionary = {"Name": newheaders}
    for dimension in range(newlatent.shape[1]):
        new_data_dictionary["z" + str(dimension)] = newlatent[:, dimension]
    # plot solid or gradient colors
    if lm.plot_is_gradient:
        if newlatent.shape[0] > 256:
            new_data_dictionary["colors"] = gencolor(newlatent.shape[0])
        else:
            new_data_dictionary["colors"] = linear_palette(
                next(twofiftysix), newlatent.shape[0]
            )
    else:
        chosen_color = next(solid_cycler)
        new_data_dictionary["colors"] = [
            chosen_color for _ in range(newlatent.shape[0])
        ]
    newsrc = ColumnDataSource(data=new_data_dictionary)
    data = p.scatter(
        "z0",
        "z1",
        source=newsrc,
        fill_color="colors",
        line_color=None,
        legend_label=event.item,
        muted_alpha=0,
        size=lm.bp_color_size[1],
    )

    # handle gradient colorpicker
    if lm.plot_is_gradient:
        # define a callback to update the gradient
        def color_change_callback(attr, old, new_color):
            newsrc.data["colors"] = gencolor(newlatent.shape[0], new_color)
            newsrc.data = dict(newsrc.data)
            # newsrc.trigger(
            #     "data", newsrc.data, newsrc.data
            # )  # explicitly trigger change if needed

        add_colorpicker_wgradient(data.glyph, color_change_callback, event.item)
    else:
        add_colorpicker(data.glyph, title=event.item)


def select_points(attr, old, new):  # outputs sequences of selection to textbox
    temp_df = lm.df
    temp_df = temp_df.reset_index(drop=True)
    temp_df = temp_df.loc[new]
    fasta.text = "\n".join(
        [str(">" + x + "\n" + y) for x, y in zip(temp_df["Name"], temp_df["Sequences"])]
    )


def select_map_points(event):  # selects sequences from hamiltonian map plot
    geo = event.geometry
    xx = geo["x"]
    yy = geo["y"]
    gridline_x = np.linspace(lm.grid_ranges[0], lm.grid_ranges[2], lm.pixels)
    gridline_y = np.linspace(lm.grid_ranges[1], lm.grid_ranges[3], lm.pixels)
    cl = np.arange(lm.pixels)
    step_size_x = gridline_x[1] - gridline_y[0]
    step_size_y = gridline_y[1] - gridline_y[0]
    x_grid = [
        cl[np.isclose(xx[x], gridline_x, atol=step_size_x)][0] for x in range(len(xx))
    ]
    y_grid = [
        cl[np.isclose(yy[x], gridline_y, atol=step_size_y)][0] for x in range(len(yy))
    ]
    rr, cc = polygon(x_grid, y_grid, shape=(lm.pixels, lm.pixels))
    coordinates = np.vstack((gridline_x[rr], gridline_y[cc]))

    lm.selected_positions = coordinates
    progress.text = "New sequences selected!"


def save_landscape_seqs(event):
    print("saving landscape selection...")
    coordinates = torch.tensor(lm.selected_positions).float().T
    generated_sequences = lm.the_model.generate_sequences(
        coordinates, argmax_sequence=True
    )
    SeqIO.write(generated_sequences, "landscape_selection.fasta", "fasta")

def save_textbox_seqs(event):
    print("saving textbox...")
    with open("textbox_sequences.fasta", "w") as f:
        f.writelines(fasta.text)

def toggle_legend(event):
    if p.legend.visible == True:
        p.legend.visible = False
        leg.label = "Turn On Legend"
    else:
        p.legend.visible = True
        leg.label = "Turn Off Legend"


def select_data(event):  # loads csv, updates column select dropdown
    lm.update_labeldf(pd.read_csv(os.path.join(data_folder, event.item)))
    lm.update_ldf_labels(lm.label_df)
    column_select.menu = lm.ldf_labels


def change_colors(event):
    # two types of data will be plotted, categorical and numerical
    # detect which type (based on #unique/#datapoints)
    # if categorical, define color column based on class
    # else, plot with colormap
    decision_ratio = len(lm.label_df[event.item].unique()) / len(
        lm.label_df[event.item]
    )
    print(decision_ratio)
    lm.update_cds_column(event.item)
    base_tooltip = p.hover[0].tooltips
    if len(base_tooltip) > 1:
        base_tooltip.pop(-1)
    base_tooltip.append((event.item + " ", "@" + event.item))
    p.hover[0].tooltips = base_tooltip

    if decision_ratio >= 0.6:
        cm = p.select_one(LinearColorMapper)
        cm.update(
            low=lm.label_df[event.item].min(),
            high=lm.label_df[event.item].max(),
        )
        lm.base_plot.glyph.fill_color = {
            "field": event.item,
            "transform": color_mapper,
        }
        lm.update_labels(["Training Data" for _ in range(len(lm.label_df[event.item]))])
        color_bar.visible = True
        update_checkbox()
    else:
        if lm.bp_color_size[0] == "white":
            color_bar.visible = True
        value_list = lm.label_df[event.item].unique().tolist()
        color_list = [next(VAE_custom_legend_pallet) for _ in range(len(value_list))]
        newcolors = {v: c for c, v in zip(color_list, value_list)}
        colored_values = [newcolors[x] for x in lm.label_df[event.item]]
        lm.update_colors(colored_values)
        lm.base_plot.glyph.fill_color = "colors"
        lm.update_labels(lm.label_df[event.item])
        update_checkbox()


def update_checkbox() -> None:  # used to initialize new data in Legend tab
    labels = lm.base_cds.data["Labels"]
    if not isinstance(labels, list):
        labels = labels.to_list()
    labels = set([str(label) for label in labels])

    the_checkbox.labels = list(labels)
    the_checkbox.active = list(range(len(labels)))


def update_checkbox_data(
    attr, old, new
) -> None:  # updates base glyph data with selected data.
    active_labels = [the_checkbox.labels[i] for i in the_checkbox.active]
    labels_to_remove = [
        the_checkbox.labels[i]
        for i in range(len(the_checkbox.labels))
        if the_checkbox.labels[i] not in active_labels
    ]
    lm.change_plot_from_selection(labels_to_remove)


def checkall(event):  # check all functionality
    if len(the_checkbox.active) == len(set(lm.unmodified_label_df["Labels"])):
        the_checkbox.active = []
    else:
        the_checkbox.active = list(range(len(set(lm.unmodified_label_df["Labels"]))))


def change_plot_type(event):
    if lm.plot_is_gradient:
        lm.plot_is_gradient = False
        solid_plot.label = "Plotting Additional MSA as Solid Color"
    else:
        lm.plot_is_gradient = True
        solid_plot.label = "Plotting Additional MSA as Gradient"


# Create a blank figure with labels
t = Title()  # ,aspect_ratio=1.0,height_policy='max'
p = figure(title=t, x_axis_label="z0", y_axis_label="z1", toolbar_location="below")
p.on_event("selectiongeometry", select_map_points)

select = LassoSelectTool(select_every_mousemove=False)
p.add_tools(HoverTool(tooltips=[("ID ", "@Name")]))
p.add_tools(select)
color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
color_bar = ColorBar(color_mapper=color_mapper, width=5, label_standoff=5)
color_bar.visible = False
p.add_layout(color_bar, "right")

# p.output_backend = 'svg' # comment me out if you want pngs

# All dropdown boxes
model_d = Dropdown(
    label="Select Model", menu=model_directory, sizing_mode="scale_width"
)
model_d.on_click(select_the_model)
msa_d = Dropdown(label="Training MSA", menu=msa_directory, sizing_mode="scale_width")
msa_d.on_click(plot_base_data)
add_d = Dropdown(label="Additional Seqs", menu=msa_directory, sizing_mode="scale_width")
add_d.on_click(plot_data)
data_select = Dropdown(
    label="Select Data CSV", menu=data_directory, sizing_mode="scale_width"
)
data_select.on_click(select_data)
column_select = Dropdown(
    label="Choose CSV column", menu=lm.ldf_labels, sizing_mode="scale_width"
)
column_select.on_click(change_colors)

# Folder selection buttons and dropdowns
model_folder_button = Button(label="Select Model Folder", button_type="primary")
msa_folder_button = Button(label="Select MSA Folder", button_type="primary")
data_folder_button = Button(label="Select Data Folder", button_type="primary")

model_folder_dropdown = Dropdown(
    label="Model Folder",
    menu=get_subfolders("."),
    sizing_mode="scale_width",
    styles={
        "background-color": "#e3f2fd",
        "color": "#0d47a1",
        "font-weight": "bold",
    },
)
msa_folder_dropdown = Dropdown(
    label="MSA Folder",
    menu=get_subfolders("."),
    sizing_mode="scale_width",
    styles={
        "background-color": "#f1f8e9",
        "color": "#33691e",
        "font-weight": "bold",
    },
)
data_folder_dropdown = Dropdown(
    label="Data Folder",
    menu=get_subfolders("."),
    sizing_mode="scale_width",
    styles={
        "background-color": "#fff3e0",
        "color": "#e65100",
        "font-weight": "bold",
    },
)


def update_model_folder(event):
    global model_folder, model_directory
    model_folder = event.item
    lm.model_folder = model_folder
    model_directory = get_files(model_folder)
    model_d.menu = model_directory


def update_msa_folder(event):
    global msa_folder, msa_directory
    msa_folder = event.item
    lm.msa_folder = msa_folder
    msa_directory = get_files(msa_folder)
    msa_d.menu = msa_directory
    add_d.menu = msa_directory


def update_data_folder(event):
    global data_folder, data_directory
    data_folder = event.item
    lm.data_folder = data_folder
    data_directory = get_files(data_folder)
    data_select.menu = data_directory


model_folder_dropdown.on_click(update_model_folder)
msa_folder_dropdown.on_click(update_msa_folder)
data_folder_dropdown.on_click(update_data_folder)

# Buttons!
leg = Button(label="Turn Off Legend", button_type="success")
leg.on_click(toggle_legend)
checker = Button(label="Un/Check All", button_type="success")
checker.on_click(checkall)
solid_plot = Button(label="Plotting Additional MSA as Gradient", button_type="success")
solid_plot.on_click(change_plot_type)
save_landscape = Button(
    label="Save lasso'd LGL generated sequences", button_type="success"
)
save_landscape.on_click(save_landscape_seqs)
save_txtbox = Button(
    label="Save sequences from the textbox below", button_type="success"
)
save_txtbox.on_click(save_textbox_seqs)


# Create checkbox for Legend Tab
the_checkbox = CheckboxGroup(labels=[])
the_checkbox.on_change("active", update_checkbox_data)

# Create initial fasta output textbox
fasta = PreText(
    text="""Displays fasta format from selection here""",
    styles={"overflow-y": "scroll", "height": "300px", "width": "600px"},
)
progress = PreText(
    text="""No sequences selected""",
    styles={"overflow-y": "scroll", "height": "300px", "width": "600px"},
)

# row grouped items
colorpickersection = row()
seq_save_section = row(save_landscape, save_txtbox)

main_layout = column(
    Div(
        text="<div style='background:#e3f2fd; color:#0d47a1; padding:4px; font-weight:bold;'>Select Model folder and trained model</div>"
    ),
    model_folder_dropdown,
    model_d,
    Div(text="<hr style='border:1px solid #90caf9; margin:8px 0;'>"),
    Div(
        text="<div style='background:#f1f8e9; color:#33691e; padding:4px; font-weight:bold;'>Select MSA folder and MSAs</div>"
    ),
    msa_folder_dropdown,
    msa_d,
    add_d,
    Div(text="<hr style='border:1px solid #a5d6a7; margin:8px 0;'>"),
    Div(
        text="<div style='background:#fff3e0; color:#e65100; padding:4px; font-weight:bold;'>Select data label folder and label spreadsheet</div>"
    ),
    data_folder_dropdown,
    data_select,
    column_select,
    Div(text="<hr style='border:1px solid #ffb74d; margin:8px 0;'>"),
    leg,
    solid_plot,
    seq_save_section,
    colorpickersection,
    fasta,
)
main_panel = TabPanel(child=main_layout, title="Main Panel")
legend_layout = column(checker, the_checkbox)
legend_panel = TabPanel(child=legend_layout, title="Legend")

# Serve plot, build layout
doc = curdoc()
doc.add_root(row(p, Tabs(tabs=[main_panel, legend_panel])))
