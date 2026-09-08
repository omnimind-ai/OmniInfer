use super::*;

#[derive(Debug)]
pub(super) struct VulkanSelection {
    pub(super) visible: Vec<String>,
    pub(super) selected: BTreeMap<String, String>,
}

pub(super) fn llama_cpp_vulkan_selection(
    backend: &backend_registry::BackendSpec,
    args: &[String],
) -> Result<Option<VulkanSelection>> {
    if cfg!(target_os = "macos")
        || backend.family != "llama.cpp"
        || !backend.id.starts_with("llama.cpp-")
        || !backend.capabilities.iter().any(|cap| cap == "vulkan")
    {
        return Ok(None);
    }
    let visible = match std::env::var("GGML_VK_VISIBLE_DEVICES") {
        Ok(value) => parse_vulkan_visible_devices(&value)?,
        Err(std::env::VarError::NotPresent) => vulkan_physical_gpu_indices()?,
        Err(error) => return Err(error.into()),
    };
    let selected = select_vulkan_devices(args, &visible)?;
    Ok(Some(VulkanSelection { visible, selected }))
}

fn parse_vulkan_visible_devices(value: &str) -> Result<Vec<String>> {
    let mut visible = Vec::new();
    for item in value
        .split(|c: char| c == ',' || c.is_ascii_whitespace())
        .filter(|x| !x.is_empty())
    {
        let index = item.parse::<u32>()?.to_string();
        if visible.contains(&index) {
            anyhow::bail!("duplicate physical device in GGML_VK_VISIBLE_DEVICES");
        }
        visible.push(index);
    }
    if visible.is_empty() {
        anyhow::bail!("GGML_VK_VISIBLE_DEVICES does not select a GPU");
    }
    Ok(visible)
}

pub(super) fn select_vulkan_devices(
    args: &[String],
    visible: &[String],
) -> Result<BTreeMap<String, String>> {
    let mut requested = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if let Some(value) = arg
            .strip_prefix("--device=")
            .or_else(|| arg.strip_prefix("-dev="))
        {
            requested = Some(value);
        } else if matches!(arg.as_str(), "--device" | "-dev") {
            index += 1;
            requested = Some(
                args.get(index)
                    .ok_or_else(|| anyhow::anyhow!("--device requires a value"))?
                    .as_str(),
            );
        }
        index += 1;
    }
    if requested == Some("none") {
        return Ok(BTreeMap::new());
    }
    let indices = match requested {
        None => (0..visible.len()).collect::<Vec<_>>(),
        Some(value) => value
            .split(',')
            .map(|name| {
                name.strip_prefix("Vulkan")
                    .ok_or_else(|| {
                        anyhow::anyhow!("expected Vulkan<index> or none in --device: {name}")
                    })?
                    .parse::<usize>()
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let mut selected = BTreeMap::new();
    for logical in indices {
        let physical = visible
            .get(logical)
            .ok_or_else(|| anyhow::anyhow!("Vulkan{logical} is not visible"))?;
        selected.insert(logical.to_string(), physical.clone());
    }
    if selected.is_empty() {
        anyhow::bail!("no Vulkan GPUs available for placement");
    }
    Ok(selected)
}

pub(super) fn vulkan_placement_budget(
    estimated: &ResourceBudget,
    selected: &BTreeMap<String, String>,
) -> Result<ResourceBudget> {
    let domains = selected
        .values()
        .cloned()
        .map(MemoryDomain::Vulkan)
        .collect::<Vec<_>>();
    if domains.is_empty() {
        return Ok(estimated.clone());
    }
    let mut components = Vec::new();
    if estimated.components().is_empty() {
        let total = estimated
            .domains()
            .values()
            .try_fold(0_u64, |total, bytes| {
                total
                    .checked_add(*bytes)
                    .ok_or_else(|| anyhow::anyhow!("Vulkan budget overflow"))
            })?;
        components.extend(assign_component("estimated_total", total, &domains, false)?);
    }
    for component in estimated.components() {
        // Actual tensor splits are unknown until native startup; provisional
        // admission reserves a ceiling on each selected device before launch.
        components.extend(assign_component(
            &component.name,
            component.bytes,
            &domains,
            false,
        )?);
    }
    ResourceBudget::from_components(components).map_err(Into::into)
}

fn vulkan_physical_gpu_indices() -> Result<Vec<String>> {
    use ash::vk;
    let entry = unsafe { ash::Entry::load() }?;
    let info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&info),
            None,
        )
    }?;
    let result = (|| -> Result<Vec<String>> {
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let mut indices = Vec::new();
        let mut uuids = Vec::new();
        for (index, device) in devices.into_iter().enumerate() {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            unsafe { instance.get_physical_device_properties2(device, &mut properties) };
            if matches!(
                properties.properties.device_type,
                vk::PhysicalDeviceType::DISCRETE_GPU | vk::PhysicalDeviceType::INTEGRATED_GPU
            ) && !uuids.contains(&id.device_uuid)
            {
                uuids.push(id.device_uuid);
                indices.push(index.to_string());
            }
        }
        if indices.is_empty() {
            anyhow::bail!("Vulkan loader reported no physical GPU");
        }
        Ok(indices)
    })();
    unsafe { instance.destroy_instance(None) };
    result
}
