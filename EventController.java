/*
 * Copyright © 2015-2026 the original author or authors.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  @Author: xxc055718@ymdd.com
 *  @LastModified: 2024-05-21 09:37:18.031
 */

package com.yl.larksuite;

import com.deepl.api.*;
import com.google.gson.internal.LinkedTreeMap;
import com.google.gson.reflect.TypeToken;
import com.lark.oapi.Client;
import com.lark.oapi.core.utils.Jsons;
import com.lark.oapi.event.EventDispatcher;
import com.lark.oapi.sdk.servlet.ext.ServletAdapter;
import com.lark.oapi.service.approval.v4.model.*;
import com.lark.oapi.service.authen.v1.model.*;
import com.lark.oapi.service.contact.v3.enums.BatchUserUserIdTypeEnum;
import com.lark.oapi.service.contact.v3.model.BatchUserReq;
import com.lark.oapi.service.contact.v3.model.BatchUserResp;
import com.lark.oapi.service.contact.v3.model.BatchUserRespBody;
import com.yl.platform.utils.StringUtil;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.*;

import com.vladsch.flexmark.html.HtmlRenderer;
import com.vladsch.flexmark.parser.Parser;

/**
 * 飞书的审批事件订阅.
 *
 * @author caiye
 * @version 1.0.0-SNAPSHOT
 * @date 2024/5/17 10:17 PM
 * @since 1.0.0
 */
@SuppressWarnings({"SerializableNonStaticInnerClassWithoutSerialVersionUID", "rawtypes", "unchecked"})
@Slf4j
@Controller
public class EventController {

	// 外置浏览器, 临时token, 一次便失效
	private static String WB_TOKEN_PREFIX = "lsp::session::wb::";
	private static String REDIS_SESSION_KEY_PREFIX = "lsp::session::";

	private String ROBOT_IMG_URL = "https://s1-imfile.feishucdn.com/static-resource/v1/v3_00b5_8df4b3b5-ca80-4c43-a01e-f64fe24f255g~?image_size=72x72&cut_type=&quality=&format=png&sticker_format=.webp";

	@Value("${larksuite.appId}")
	private String appId;
	@Value("${translate.url}")
	private String translateUrl;

	// 1. 注入消息处理器
	private final EventDispatcher eventDispatcher;
	// 2. 注入 ServletAdapter 实例
	private final ServletAdapter servletAdapter;

	private final Client larksuiteClient;

	private final Translator translator;

	private final HtmlRenderer markdownRender;

	private final Parser markdownParser;

	private final RedisTemplate<String, CreateAccessTokenRespBody> redisTemplate;

	private final RedisTemplate<String, String> stringRedisTemplate;

	public EventController(EventDispatcher eventDispatcher, ServletAdapter servletAdapter, Client larksuiteClient,
	                       Translator translator, HtmlRenderer markdownRender, Parser markdownParser,
	                       RedisTemplate redisTemplate, RedisTemplate<String, String> stringRedisTemplate) {
		this.eventDispatcher = eventDispatcher;
		this.servletAdapter = servletAdapter;
		this.larksuiteClient = larksuiteClient;
		this.translator = translator;
		this.markdownRender = markdownRender;
		this.markdownParser = markdownParser;
		this.redisTemplate = redisTemplate;
		this.stringRedisTemplate = stringRedisTemplate;
	}

	/**
	 * 飞书事件调用接口
	 * 回调扩展包提供的事件回调处理器，并以 15秒、5分钟、1小时、6小时 的间隔重新推送事件，最多重试 4 次
	 * 事件示例值：
	 * 1、用来测试服务是否可用
	 * {"challenge":"982ffe0b-0132-4dbe-89f1-5913df39914e","token":"FDCbX4kmYvRCX988avejvdqM6FG74Z23","type":"url_verification"}
	 * 2、真实的审批实例创建事件示例
	 * {"uuid": "a76bfcd854e35ec88bc6ef98fa4d23c0", "event": {"app_id": "cli_a6cf4d56fe78d013", "approval_code": "4EB6BDFC-5E35-4025-86AD-BEE010364C3F", "instance_code": "542248C3-11AA-4BE1-BBEB-51AFC87FF9FD", "instance_operate_time": "1716174311566", "operate_time": "1716174311566", "status": "PENDING", "tenant_key": "15206d1761c11740", "type": "approval_instance", "uuid": "506b065f-fb69-445d-bb48-7a48c619910b"}, "token": "FDCbX4kmYvRCX988avejvdqM6FG74Z23", "ts": "1716174328.962266", "type": "event_callback"}
	 */
	@PostMapping("/webhook/event")
	@ResponseBody
	public void event(HttpServletRequest request, HttpServletResponse response) throws Throwable {
		servletAdapter.handleEvent(request, response, eventDispatcher);
	}

	/**
	 * 飞书管理后台配置订阅事件后，还得发送订阅指定的审批的事件   可以使用
	 * @param approvalCode 审批定义编码, 示例值："C77212BB-08C5-414F-ACE0-1F84059FEAB9"
	 */
	@GetMapping("/approvals/{approvalCode}/subscribe")
	@ResponseBody
	public SubscribeApprovalResp subscribe(@PathVariable(name = "approvalCode") String approvalCode) throws Throwable {
		SubscribeApprovalReq subscribeApprovalReq = SubscribeApprovalReq.newBuilder()
				.approvalCode(approvalCode)
				.build();
		return larksuiteClient.approval().approval().subscribe(subscribeApprovalReq);
	}

	/**
	 * 查询审批定义
	 * @param approvalCode 审批定义编码, 示例值："C77212BB-08C5-414F-ACE0-1F84059FEAB9"
	 */
	@GetMapping("/approvals/{approvalCode}")
	@ResponseBody
	public GetApprovalResp approvalDefinition(@PathVariable(name = "approvalCode") String approvalCode) throws Throwable {
		GetApprovalReq getApprovalReq = GetApprovalReq.newBuilder()
				.approvalCode(approvalCode)
				.locale("zh-CN")
				.build();
		return larksuiteClient.approval().approval().get(getApprovalReq);
	}

	/**
	 * 查询审批实例
	 * @param instanceCode 审批实例Code, 示例值："490E42A1-9D30-4F54-9318-5A0EEADCBD5E"
	 */
	@GetMapping("/approvals/instance/{instanceCode}")
	@ResponseBody
	public GetInstanceResp approvalInstance(@PathVariable(name = "instanceCode") String instanceCode) throws Throwable {
		GetInstanceReq getInstanceReq = GetInstanceReq.newBuilder()
				.instanceId(instanceCode)
				.locale("zh-CN")
				.build();
		return larksuiteClient.approval().instance().get(getInstanceReq);
	}

	/**
	 * 显示中文主页, 无session, 则跳到授权页面
	 */
	@GetMapping(value = {"/translate"})
	public ModelAndView index(@RequestParam(name = "instanceCode") String instanceCode, String wbToken, HttpServletRequest request) throws Throwable {
		ModelAndView modelAndView = new ModelAndView();
		modelAndView.addObject("instance_code", instanceCode);

		System.out.println(request.getSession().getId());

		// 移动基于APP自带html翻译，直接返回即可
		String ua = request.getHeader("user-agent");
		if (ua.contains("Android") || ua.contains("iPhone OS")) {
			// 2、已登陆用户，显示翻译页面
			CreateAccessTokenRespBody object = redisTemplate.opsForValue().get(getRedisSessionKey(request.getSession().getId()));
			if (object != null) {
				Map<String, Object> map = translateInstance(instanceCode);
				modelAndView.addObject("data", map);
				modelAndView.setViewName("index");
			} else {
				// 1、先授权免登
				modelAndView.addObject("appId", appId);
				modelAndView.addObject("translatorUrl", Beans.getLkTargetUrl(translateUrl, instanceCode));
				modelAndView.setViewName("auth");
			}
		} else {
			// 3、pc 端基于外置浏览器html翻译, 所以先换个临时token，再用 window.open("..?lk_jump_to_browser=true") 打开外置浏览器
			if (wbToken != null && stringRedisTemplate.opsForValue().get(getWbKey(wbToken)) != null) {
				Map<String, Object> map = translateInstance(instanceCode);
				modelAndView.addObject("data", map);
				modelAndView.setViewName("index_pc");
				// 只能用一次
				stringRedisTemplate.delete(getWbKey(wbToken));
			} else {
				// 已登陆用户
				CreateAccessTokenRespBody object = redisTemplate.opsForValue().get(getRedisSessionKey(request.getSession().getId()));
				if (object != null) {
					// 2、临时token，供PC端浏览器翻译用，一次失效
					String uuid = UUID.randomUUID().toString();
					stringRedisTemplate.opsForValue().set(getWbKey(uuid), "1");

					String url = Beans.getLkTargetUrl(translateUrl, instanceCode) + "&lk_jump_to_browser=true" + "&wbToken=" + uuid;
					modelAndView.addObject("translatorUrl", url);
					modelAndView.setViewName("index_pc_pre");
				} else {
					// 1、先授权免登
					modelAndView.addObject("appId", appId);
					modelAndView.addObject("translatorUrl", Beans.getLkTargetUrl(translateUrl, instanceCode));
					modelAndView.setViewName("auth");
				}
			}
		}
		return modelAndView;
	}

	/**
	 * 当前app登录用户, 并且设置session
	 * @param code 授权码
	 */
	@GetMapping("/authentication/{code}")
	@ResponseBody
	public Object authentication(@PathVariable(name = "code") String code, HttpSession session) throws Exception {
		CreateAccessTokenReqBody createAccessTokenReqBody = CreateAccessTokenReqBody.newBuilder()
				.code(code).grantType("authorization_code").build();
		CreateAccessTokenReq createAccessTokenReq = CreateAccessTokenReq.newBuilder()
				.createAccessTokenReqBody(createAccessTokenReqBody).build();
		CreateAccessTokenResp createAccessTokenResp = larksuiteClient.authen().accessToken().create(createAccessTokenReq);

		log.info("临时授权码请求参数: {}, 当前用户响应结果: {}", Jsons.DEFAULT.toJson(createAccessTokenReq), Jsons.DEFAULT.toJson(createAccessTokenResp));
		if (createAccessTokenResp.getCode() == 0) {
			// 当前app登录用户信息、token等敏感信息
			CreateAccessTokenRespBody createAccessTokenRespBody = createAccessTokenResp.getData();
			// 不带敏感信息
			/*GetUserInfoResp getUserInfoResp = larksuiteClient.authen().userInfo().get(RequestOptions.newBuilder().userAccessToken(createAccessTokenRespBody.getAccessToken()).build());
			GetUserInfoRespBody getUserInfoRespBody = getUserInfoResp.getData();
			return getUserInfoRespBody;*/
			// 直接脱敏吧, 少一次请求
			createAccessTokenRespBody.setAccessToken(null);
			createAccessTokenRespBody.setRefreshToken(null);
			redisTemplate.opsForValue().set(getRedisSessionKey(session.getId()), createAccessTokenRespBody, Duration.ofDays(5));
			return createAccessTokenRespBody;
		}
		return null;
	}

	/**
	 * 显示中文，实际上就是拼接div(免鉴权)
	 * @param instanceCode 审批实例Code, 示例值："490E42A1-9D30-4F54-9318-5A0EEADCBD5E"
	 */
	public Map<String, Object> translateInstance(String instanceCode) throws Throwable {
		Map<String, Object> map = new HashMap<>();

		// 查询审批实例
		GetInstanceResp getInstanceResp = approvalInstance(instanceCode);
		if (getInstanceResp.getCode() == 0) {
			// todo, 是否还要判断当前用户和审批定义中的节点用户校验, 以后再说
			/*String approvalCode = getInstanceResp.getData().getApprovalCode();
			GetApprovalResp getApprovalResp = approvalDefinition(approvalCode);
			if (getApprovalResp.getCode() == 0) {
				GetApprovalRespBody getApprovalRespBody = getApprovalResp.getData();
				List<Widget> widgets = Jsons.DEFAULT.fromJson(getApprovalRespBody.getForm(), new TypeToken<List<User>>() {}.getType());
			} else {
				map.put("err", "没有找到该审批实例的审批定义数据, 审批实例编码: " + instanceCode + ", 审批定义编码: " + approvalCode + "!");
			}
			*/
			GetInstanceRespBody getInstanceRespBody = getInstanceResp.getData();
			Map<String, com.lark.oapi.service.contact.v3.model.User> users = batchUser(getInstanceRespBody);
			map.put("instance", getInstanceResp.getData());
			map.put("users", users);
			map.put("div", deepL(getInstanceRespBody, users));
		} else {
			map.put("div", "没有找到该审批实例的数据, 审批实例编码: " + instanceCode + "!");
		}
		return map;
	}

	// 获取审批人、评论人的用户信息
	public Map<String, com.lark.oapi.service.contact.v3.model.User> batchUser(GetInstanceRespBody getInstanceRespBody) throws Exception {
		Map<String, com.lark.oapi.service.contact.v3.model.User> map = new HashMap<>();
		List<String> userIds = new ArrayList<>();
		userIds.add(getInstanceRespBody.getUserId());

		InstanceTask[] instanceTasks = getInstanceRespBody.getTaskList();
		for (InstanceTask instanceTask : instanceTasks) {
			addUserId(userIds, instanceTask.getUserId());
		}

		InstanceTimeline[] instanceTimelines = getInstanceRespBody.getTimeline();
		for (InstanceTimeline instanceTimeline : instanceTimelines) {
			addUserId(userIds, instanceTimeline.getUserId());
			if ("CC".equalsIgnoreCase(instanceTimeline.getType()) && instanceTimeline.getUserIdList() != null) {
				String[] userIdList = instanceTimeline.getUserIdList();
				for (String uid : userIdList) {
					addUserId(userIds, uid);
				}
			}
		}

		InstanceComment[] instanceComments = getInstanceRespBody.getCommentList();
		for (InstanceComment instanceComment : instanceComments) {
			addUserId(userIds, instanceComment.getUserId());
		}

		BatchUserReq batchUserReq = BatchUserReq.newBuilder().
				userIds(userIds.toArray(new String[userIds.size()]))
				.userIdType(BatchUserUserIdTypeEnum.USER_ID)
				.build();
		BatchUserResp batchUserResp = larksuiteClient.contact().user().batch(batchUserReq);
		BatchUserRespBody batchUserRespBody = batchUserResp.getData();
		if (batchUserRespBody != null && batchUserRespBody.getItems() != null) {
			com.lark.oapi.service.contact.v3.model.User[] users = batchUserRespBody.getItems();
			for (com.lark.oapi.service.contact.v3.model.User user : users) {
				map.put(user.getUserId(), user);
			}
		}
		return map;
	}

	public List<String> addUserId(List<String> userIds, String userId) {
		if (StringUtil.isNotBlank(userId)) {
			userIds.add(userId);
		}
		return userIds;
	}

	public String deepL(GetInstanceRespBody getInstanceRespBody, Map<String, com.lark.oapi.service.contact.v3.model.User> users) throws DeepLException, InterruptedException {
		String container = "<div class='container-fluid'>\n"
				+ toApprovalDiv(getInstanceRespBody, users)
				+ toInstanceDiv(getInstanceRespBody)
				+ toTimelineDiv(getInstanceRespBody, users)
				+ toCommentDiv(getInstanceRespBody, users)
				+ "  <div class='row-fluid serial-num'>\n"
				+ "    编号：" + getInstanceRespBody.getSerialNumber() + "\n"
				+ "  </div>\n"
				+"</div>";

//		TextTranslationOptions textTranslationOptions = new TextTranslationOptions();
//		textTranslationOptions.setTagHandling("html");
//		String text = translator.translateText(container, LanguageCode.English, LanguageCode.Chinese, textTranslationOptions).getText();
//		log.info("翻译前: {}, 翻译后: {}", container, text);
//		return text;
		return container;
	}

	/**
	 * 审批定义div
	 */
	public Map<String, String> statusLabelMap = new HashMap<String, String>(){{
		// 通用的
		put("PENDING", "审批中");
		put("APPROVED", "已通过");
		put("REJECTED", "已拒绝");
		// 审批特有的
		put("CANCELED", "已撤回");
		put("DELETED", "已删除");
		// 审批任务节点特有的
		put("TRANSFERRED", "已转交");
		put("DONE", "已完成");
	}};
	public Map<String, String> statusColorMap = new HashMap<String, String>(){{
		// 通用的
		put("PENDING", "value-bg-info");
		put("APPROVED", "value-bg-success");
		put("REJECTED", "value-bg-danger");
		// 审批特有的
		put("CANCELED", "value-bg-warning");
		put("DELETED", "value-bg-danger");
		// 审批任务节点特有的
		put("TRANSFERRED", "value-bg-success");
		put("DONE", "value-bg-success");
	}};
	public String toApprovalDiv(GetInstanceRespBody instance, Map<String, com.lark.oapi.service.contact.v3.model.User> users) {
		String row =
				"      <div class='row-fluid row-fluid-container'>\n" +
				"        <div class='span12'>\n" +
				"          <div class='page-header form-check form-check-inline'>\n" +
				"            <div><h4>${approvalName}</h4></div>\n" +
				"            <div><span class='label value ${color} timeline2-detail-user'>${approvalStatus}</span></div>\n" +
				"          </div>\n" +
				"          <div class='row-fluid'>\n" +
				"            <div class='span12'>\n" +
				"              <img class='img_div_small' src='${userAvatarUrl}'/>\n" +
				"              <span class='label value'>${userName}</span>\n" +
				"            </div>\n" +
				"          </div>\n" +
				"          <div class='row-fluid'>\n" +
				"            <div class='span12 value' style='margin-top: 12px;'>\n" +
				"              <span>提交于 ${createTime}</span>\n" +
				"            </div>\n" +
				"          </div>\n" +
				"        </div>\n" +
				"      </div>\n";
		String createTime = dateFormat(instance.getStartTime(), "M月d日 HH:mm");
		String statusName = statusLabelMap.getOrDefault(instance.getStatus(), "审批中");
		String statusColor = statusColorMap.getOrDefault(instance.getStatus(), "value-bg-info");
		String userName = users.get(instance.getUserId()).getName();
		String userAvatarUrl = users.get(instance.getUserId()).getAvatar().getAvatar72();
		row = row.replace("${approvalName}", instance.getApprovalName())
				.replace("${color}", statusColor)
				.replace("${approvalStatus}", statusName)
				.replace("${userAvatarUrl}", userAvatarUrl)
				.replace("${userName}", userName)
				.replace("${createTime}", createTime);
		return row;
	}

	/**
	 * 审批详情div
	 */
	public String toInstanceDiv(GetInstanceRespBody instance) {
		StringBuilder sb = new StringBuilder();
		sb.append(
				"      <div class='row-fluid row-fluid-container'>\n" +
						"        <div class='span12'>\n" +
						"          <div class='page-header'>\n" +
						"            <h5>审批详情</h5>\n" +
						"          </div>\n" +
						"        </div>\n");

		String row =
				"          <div class='row-fluid'>\n" +
						"            <div class='span6 label' style='display: ${display}'>\n" +
						"              <span>${label}</span>\n" +
						"            </div>\n" +
						"            <div class='span6 value ${textClass}'>\n" +
						"              <span>${value}</span>\n" +
						"            </div>\n" +
						"          </div>\n";
		List<Widget> widgets = Jsons.DEFAULT.fromJson(instance.getForm(), new TypeToken<List<Widget>>() {}.getType());
		for (Widget widget : widgets) {
			String textClass = "";
			String value = "";
			String display = "block";
			// todo 以后改为先判断标签类型，在判断数据类型, 除非能抓取到飞书渲染后的html
			if (widget.getValue() != null) {
				if (widget.getValue() instanceof Double) {
					value = formatDoubleWithComma((Double) widget.getValue());
					if ("amount".equalsIgnoreCase(widget.getType())) {
						try {
							value = value + " " + ((LinkedTreeMap) widget.getExt()).get("currency");
						} catch (Exception e) {
							log.error(e.getMessage(), e);
						}
					}
				} else if (widget.getValue() instanceof Integer) {
					value = Integer.toString((Integer) widget.getValue());
				} else if (widget.getValue() instanceof ArrayList) {
					List<String> list = (List<String>) widget.getValue();
					if ("attachmentV2".equalsIgnoreCase(widget.getType()) || "attachment".equalsIgnoreCase(widget.getType())) {
						String[] exts = ((String)widget.getExt()).split(",");
						for (int i = 0; i < exts.length; i++) {
							value += "<div class='attachmentV2'><a href='" + list.get(i) +"'>" + exts[i] + "</a></div>";
						}
					} else if ("imageV2".equalsIgnoreCase(widget.getType()) || "image".equalsIgnoreCase(widget.getType())) {
						for (int i = 0; i < list.size(); i++) {
							value += "<img class='img_div' src='" + list.get(i) +"'/>";
						}
					} else if ("fieldList".equalsIgnoreCase(widget.getType())) {
						textClass = "text-css";
						try {
							// todo 后续改为遍历
							LinkedTreeMap ext = (LinkedTreeMap) ((ArrayList) widget.getExt()).get(0);
							String extId = (String) ext.get("id");
							String extLabel = widget.getName();
							String extValue = (String) ext.get("value");
							List<LinkedTreeMap> linkedTreeMaps = (List<LinkedTreeMap>) ((ArrayList) widget.getValue()).get(0);
							for (LinkedTreeMap linkedTreeMap : linkedTreeMaps) {
								String id = (String) linkedTreeMap.get("id");
								String name = (String) linkedTreeMap.get("name");
								String value1 = "";
								if (linkedTreeMap.get("value") instanceof Double) {
									value1 = String.format("%.2f", (Double)linkedTreeMap.get("value"));
								} else if (linkedTreeMap.get("value") instanceof Integer)  {
									value1 = Integer.toString((Integer) linkedTreeMap.get("value") );
								} else {
									value1 = (String)linkedTreeMap.get("value");
								}
								value += row.replace("${label}", name)
										.replace("${value}", value1)
										.replace("${textClass}", "")
										.replace("${display}", display);
								if (extId != null && extId.equals(id)) {
									extLabel += " 汇总 - " + name;
								}
							}
							String rowContext =
									"          <div class='row-fluid'>\n" +
											"            <div class='span6 label' style='display: ${display}'>\n" +
											"              <span>${label}</span>\n" +
											"            </div>\n" +
											"            <div class='span6 value ${textClass}'>\n" +
											"               ${value}" +
											"            </div>\n" +
											"          </div>\n";
							rowContext = rowContext.replace("${label}", widget.getName())
									.replace("${value}", value)
									.replace("${textClass}", textClass)
									.replace("${display}", display);
							String extRowContent = row.replace("${label}", extLabel)
									.replace("${value}", extValue)
									.replace("${textClass}", "")
									.replace("${display}", display);
							sb.append(rowContext).append(extRowContent);
							// 这个判断分支，到这里结束
							continue;
						} catch (Exception e) {
							value = StringUtil.join(list, ",");
							log.error(e.getMessage(), e);
						}
					} else {
						value = StringUtil.join(list, ",");
					}
				} else {
					if ("date".equalsIgnoreCase(widget.getType())) {
//						int timezoneOffsetMinutes = widget.getTimezoneOffset();
//						ZoneOffset zoneOffset = ZoneOffset.ofTotalSeconds(timezoneOffsetMinutes * 60);
						// 解析时间和时区
						OffsetDateTime dateTime = OffsetDateTime.parse((String)widget.getValue());
						// 创建 ZoneOffset 对象
						DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
						value = dateTime.format(formatter);
					}  else if ("text".equalsIgnoreCase(widget.getType()) || "textarea".equalsIgnoreCase(widget.getType())) {
						display = "none";
						textClass = "text-css";
						value = (String)widget.getValue();
						value = value.replace("\n", "<br />");
					} else {
						value = (String)widget.getValue();
					}
				}
			}

			String rowContext = row.replace("${label}", widget.getName())
					.replace("${value}", value)
					.replace("${textClass}", textClass)
					.replace("${display}", display);
			sb.append(rowContext);
		}
		sb.append("      </div>");
		return sb.toString();
	}

	/**
	 * 审批记录div
	 */
	public Map<String, String> statusLabelMapTimeline = new HashMap<String, String>(){{
		// 通用的
		put("PENDING", "审批中");
		put("APPROVED", "已同意");
		put("REJECTED", "已拒绝");
		// 审批特有的
		put("CANCELED", "已撤回");
		put("DELETED", "已删除");
		// 审批任务节点特有的
		put("TRANSFERRED", "已转交");
		put("DONE", "已完成");
	}};
	public String toTimelineDiv(GetInstanceRespBody instance, Map<String, com.lark.oapi.service.contact.v3.model.User> users) {
		StringBuilder sb = new StringBuilder();
		sb.append(
				"      <div class='row-fluid row-fluid-container'>\n" +
						"        <div class='span12'>\n" +
						"          <div class='page-header'>\n" +
						"            <h5>审批记录</h5>\n" +
						"          </div>\n" +
						"          <div class='row-fluid'>\n" +
						"            <div class='span12'>\n" +
						"              <div class='timeline2-centered'>\n");
		String row =
				"                <div class='timeline2-entry'>\n" +
						"                  <div class='timeline2-entry-inner'>\n" +
						"                    <div class='timeline2-icon ${iconColor}'></div>\n" +
						"                    <div class='timeline2-label'>\n" +
						"                      <h2>${action}</h2>\n" +
						"                      <div class='form-check form-check-inline'>\n" +
						"                        <img class='img_div_std' src='${userAvatarUrl}'/>\n" +
						"                        <div class='timeline2-detail-user'> \n" +
						"                          <div><span class='label value timeline2-detail-user'>${userName}</span></div>\n" +
						"                          <div><span class='label value ${color} timeline2-detail-user'>${userAction}</span></div>\n" +
						"                        </div>\n" +
						"                        <div class='timeline2-detail-date'>\n" +
						"                          <div><span class='label value timeline2-detail-date'>${dateBefore}天前</span></div>\n" +
						"                          <div><span class='label value timeline2-detail-date'>${createTime}</span></div>\n" +
						"                        </div>\n" +
						"                      </div>\n" +
						"                    </div>\n" +
						"                  </div>\n" +
						"                </div>\n";


		Map<String, InstanceTask> taskMap = new HashMap<>();
		InstanceTask[] instanceTasks = instance.getTaskList();
		for (InstanceTask instanceTask : instanceTasks) {
			taskMap.put(instanceTask.getId(), instanceTask);
		}

		InstanceTimeline[] timelines = instance.getTimeline();
		for (InstanceTimeline timeline : timelines) {
			com.lark.oapi.service.contact.v3.model.User user = users.get(timeline.getUserId());
			String createTime = dateFormat(timeline.getCreateTime(), "M月d日 HH:mm");
			Long dateBefore = dateBefore(timeline.getCreateTime());
			if ("start".equalsIgnoreCase(timeline.getType())) {
				sb.append(row.replace("${iconColor}", "bg-success")
						.replace("${action}", "提交")
						.replace("${userAvatarUrl}", user.getAvatar().getAvatar72())
						.replace("${userName}", user.getName())
						.replace("${color}", "value-bg-success")
						.replace("${userAction}", "已提交")
						.replace("${dateBefore}", dateBefore.toString())
						.replace("${createTime}", createTime)
				);
			} else if ("CC".equalsIgnoreCase(timeline.getType())) {
				sb.append(row.replace("${iconColor}", "bg-success")
						.replace("${action}", "抄送")
						.replace("${userAvatarUrl}", ROBOT_IMG_URL)
						.replace("${userName}", "系统")
						.replace("${color}", "value-bg-secondary ")
						.replace("${userAction}", "已转交")
						.replace("${dateBefore}", dateBefore.toString())
						.replace("${createTime}", createTime)
				);
			} else {
				InstanceTask instanceTask = taskMap.get(timeline.getTaskId());
				String nodeName = instanceTask.getNodeName();
				String color = statusColorMap.get(instanceTask.getStatus());
				String iconColor = color.replace("value-", "");
				String userAction = statusLabelMapTimeline.get(instanceTask.getStatus());
				sb.append(row.replace("${iconColor}", iconColor)
						.replace("${action}", nodeName)
						.replace("${userAvatarUrl}", user.getAvatar().getAvatar72())
						.replace("${userName}", user.getName())
						.replace("${color}", color)
						.replace("${userAction}", userAction)
						.replace("${dateBefore}", dateBefore.toString())
						.replace("${createTime}", createTime)
				);
			}
		}

		sb.append(
				"                <div class='timeline2-entry begin'>\n" +
						"                  <div class='timeline2-entry-inner'>\n" +
						"                    <div class='timeline2-icon bg-secondary'></div>\n" +
						"                    <div class='timeline2-label'>\n" +
						"                      <h2>结束</h2>\n" +
						"                      <div class='form-check form-check-inline'>\n" +
						"                        <img class='img_div_std' src='" + ROBOT_IMG_URL + "'/>\n" +
						"                        <div class='timeline2-detail-user' style='display:grid;'> \n" +
						"                          <div><span class='label value timeline2-detail-user timeline2-detail-user-end'>系统</span></div>\n" +
						"                          <div><span class='label value timeline2-detail-user timeline2-detail-user-end1'>${finish}</span></div>\n" +
						"                        </div>\n" +
						"                      </div>\n" +
						"                    </div>\n" +
						"                  </div>\n" +
						"                </div>" +
						"              </div>\n" +
						"            </div>\n" +
						"          </div>\n" +
						"        </div>\n" +
						"      </div>\n");
		String div = sb.toString();
		if ("PENDING".equalsIgnoreCase(instance.getStatus())) {
			div = div.replace("${finish}", "未结束");
		} else {
			div = div.replace("${finish}", "已结束");
		}
		return div;
	}

	/**
	 * 评论div
	 */
	public String toCommentDiv(GetInstanceRespBody instance, Map<String, com.lark.oapi.service.contact.v3.model.User> users) {
		StringBuilder sb = new StringBuilder();
		sb.append(
				"      <div class='row-fluid row-fluid-container'>\n" +
						"        <div class='span12'>\n" +
						"          <div class='page-header'>\n" +
						"            <h5>全文评论</h5>\n" +
						"          </div>\n"
		);
		String row =
				"          <div class='row-fluid'>\n" +
						"            <div class='form-check form-check-inline' style='align-items: start'>\n" +
						"              <div><img class='img_div_std' src='${userAvatarUrl}' /></div>\n" +
						"              <div>\n" +
						"                <div class='form-check form-check-inline'>\n" +
						"                  <div class='timeline2-detail-user'>\n" +
						"                    <span class='label value comment-user'>${userName}</span>\n" +
						"                  </div>"+
						"                  <div class='timeline2-detail-date'>\n" +
						"                    <span class='label value comment-date'>${createTime}</span>\n" +
						"                  </div>\n" +
						"                </div>\n" +
						"                <div><span class='label value comment-detail'>${comment}</span></div>\n" +
						"              </div>\n" +
						"            </div>\n" +
						"          </div>\n";

		InstanceComment[] comments = instance.getCommentList();
		for (InstanceComment comment : comments) {
			com.lark.oapi.service.contact.v3.model.User user = users.get(comment.getUserId());
			String commentDetail = comment.getComment().replace("\n", "<br />");
			String html = markdownRender.render(markdownParser.parse(commentDetail));
			sb.append(row.replace("${userAvatarUrl}", user.getAvatar().getAvatar72())
					.replace("${userName}", user.getName())
					.replace("${createTime}", dateFormat(comment.getCreateTime(), "M月d日 HH:mm"))
					.replace("${comment}", html)
			);
		}
		sb.append(
				"        </div>\n" +
				"      </div>"
		);
		return sb.toString();
	}

	public static String getWbKey (String uuid) {
		return WB_TOKEN_PREFIX + uuid;
	}
	public static String getRedisSessionKey (String sessionId) {
		return REDIS_SESSION_KEY_PREFIX + sessionId;
	}

	public static String dateFormat(String timestamp, String format) {
		Instant instant = Instant.ofEpochMilli(Long.valueOf(timestamp));
		ZonedDateTime zdt = instant.atZone(ZoneId.systemDefault());
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
		String formattedDate = zdt.format(formatter);
		return formattedDate;
	}

	public static Long dateBefore(String timestamp) {
		Instant instant = Instant.ofEpochMilli(Long.valueOf(timestamp));
		LocalDate date1 = instant.atZone(ZoneId.systemDefault()).toLocalDate();
		LocalDate date2 = Instant.now().atZone(ZoneId.systemDefault()).toLocalDate();
		return ChronoUnit.DAYS.between(date1, date2);
	}

	public static String formatDoubleWithComma(double value) {
		if (value < 0) {
			return "-" + formatDoubleWithComma(-value);
		}

		String strValue = String.format("%.2f", value); // 保留两位小数，根据需要调整
		String integerPart = strValue.split("\\.")[0];
		String decimalPart = strValue.split("\\.")[1];

		StringBuilder sb = new StringBuilder();
		for (int i = integerPart.length() - 1, j = 1; i >= 0; i--, j++) {
			sb.append(integerPart.charAt(i));
			if (j % 3 == 0) {
				sb.append(",");
			}
		}

		// 反转字符串以得到正确的顺序
		sb.reverse();

		// 添加小数部分
		sb.append(".").append(decimalPart);

		String r = sb.toString();
		if (r.startsWith(",")) {
			r = r.substring(1);
		}
		return r;
	}

	public static void main(String[] args) {
		String str = formatDoubleWithComma(14000000d);
		System.out.println(str);

		Parser parser = Parser.builder().build();
		HtmlRenderer renderer = HtmlRenderer.builder().nodeRendererFactory(new Beans.CustomParagraphRenderer.Factory()).build();

		// 使用解析器和渲染器将Markdown转换为HTML
		String html1 = renderer.render(parser.parse("JOG - PT. SUKARNO PUTRO TRANS\\nPayment Application for Ritase Car Rent\\nPeriod April 2024\\nExternal Route：JOG-SOC-DPS\\n\\nNon Proportion\\n1 OW CDD Long, @Rp 6.000.000, Total= Rp 6.000.000\\n Proportion\\n1 PP CDD Long, @Rp 8.200.000, Total= Rp 8.200.000\\n\\nTOTAL = Rp 14.200.000\\n\\nAgreement Period untill 11 April 2024\\n\\nNote for Finance HQ : \\nData in Capacity Platform JFS, JOG-DPS Difference Rp - 2.200.000"));
		System.out.println(html1);
		System.out.println(html1.replace("\\n", "</br>"));

		String html = renderer.render(parser.parse("[aaaa](http://106.54.227.8:9001/larksuite/translate?instanceCode=490E42A1-9D30-4F54-9318-5A0EEADCBD5E)"));
		System.out.println(html);

		String json = "[{\"id\":\"widget16987224160410001\",\"name\":\"供应商名称 - Vendor Name\",\"type\":\"input\",\"ext\":null,\"value\":\"JOHN ENROE PARLINDUNGAN\"},{\"id\":\"widget16987224541650001\",\"name\":\"费用说明 - Expense Description\",\"type\":\"textarea\",\"ext\":null,\"value\":\"PLACE RENTAL FEE\\nOutlet : GW MEDAN\\nSITE AREA : PERCUT SEI TUAN, SUMATERA UTARA \\nLAND AREA : 756 M2  \\nBUILDING AREA : 796 M2\\nCONTRACT PERIODE : 20 MAY 2024 - 19 MAY 2025\\nRENT (EXC TAX) : Rp 180.000.000\\nRENT (INC TAX) : Rp 200.000.000\\nDEPOSIT :  Rp 20.000.000\\n\\nNOTE : FOR 2 UNITS \\nUnit : Pergudangan Intan blok 99-R\"},{\"id\":\"widget16987239235610001\",\"name\":\"总付款（自动计算）- Total Payment (auto calculate)\",\"type\":\"formula\",\"ext\":{\"capitalValue\":\"贰亿元整\"},\"value\":200000000},{\"id\":\"widget16987224372640001\",\"name\":\"用于接收单据的供应商电子邮件 - Vendor Email for Receiving Bukti Potong\",\"type\":\"textarea\",\"ext\":null,\"value\":\"wilsonyohan@gmail.com\"},{\"id\":\"widget16987224684130001\",\"name\":\"区域名称 - Area\",\"type\":\"radioV2\",\"ext\":null,\"value\":\"MES - 棉兰\",\"option\":{\"key\":\"lodrhxe2-6fmodkoa134-2\",\"text\":\"MES - 棉兰\"}},{\"id\":\"widget16510509268920001\",\"name\":\"付款类别 - Payment Category\",\"type\":\"radioV2\",\"ext\":null,\"value\":\"直营 - Ops\",\"option\":{\"key\":\"lodrhxe2-ucx8xu7461j-28\",\"text\":\"直营 - Ops\"}},{\"id\":\"widget16992801098670001\",\"name\":\"说明 2\",\"type\":\"text\",\"ext\":null,\"value\":\"1.Petty Cash OPS Reimbursement:pls select from payment type OPS\\n2.Petty Cash DSO Reimbursement:pls select from payment type NWK\\n3.Petty Cash Office Reimbursement\"},{\"id\":\"widget16987333173110001\",\"name\":\"付款类型 OPS\",\"type\":\"radioV2\",\"ext\":null,\"value\":\"Lokasi Gateway 房源\",\"option\":{\"key\":\"lodxydpz-h83dxj50al-1\",\"text\":\"Lokasi Gateway 房源\"}},{\"id\":\"widget16987326168340001\",\"name\":\"GW Code - 分拨编码\",\"type\":\"radioV2\",\"ext\":null,\"value\":\"MES99A/GW-Medan\",\"option\":{\"key\":\"lodxjfyo-nlkhkv5gfvo-21\",\"text\":\"MES99A/GW-Medan\"}},{\"id\":\"widget17005716459680001\",\"name\":\"Description 3\",\"type\":\"text\",\"ext\":null,\"value\":\"Untuk FTL Details , harap upload file dalam bentuk excel. \"},{\"id\":\"widget16510510138590001\",\"name\":\"Invoice Date - 发票日期\",\"type\":\"date\",\"ext\":null,\"value\":\"2024-05-22T01:00:00+08:00\",\"timezoneOffset\":-420},{\"id\":\"widget16510510048490001\",\"name\":\"Invoice Number - 发票号码\",\"type\":\"input\",\"ext\":null,\"value\":\"003/WP-INV/V/2024\"},{\"id\":\"widget17055647946040001\",\"name\":\"Period - 日期\",\"type\":\"input\",\"ext\":null,\"value\":\"2405\"},{\"id\":\"widget17055648161900001\",\"name\":\"说明 4\",\"type\":\"text\",\"ext\":null,\"value\":\"例如，对于2023年12月1日至2023年12月31日期间，请输入期间“2312”\"},{\"id\":\"widget16987227752500001\",\"name\":\"Nomor Faktur Pajak -  税务发票号码\",\"type\":\"input\",\"ext\":null,\"value\":\"0\"},{\"id\":\"widget16510510254730001\",\"name\":\"DPP Amount - DPP 金额\",\"type\":\"amount\",\"ext\":{\"capitalValue\":\"贰亿贰仟万元整\",\"currency\":\"IDR\",\"currencyRange\":[\"IDR\",\"CNY\",\"USD\"],\"maxValue\":\"\",\"minValue\":\"\"},\"value\":220000000},{\"id\":\"widget16987230786410001\",\"name\":\"PPN\",\"type\":\"amount\",\"ext\":{\"capitalValue\":\"零\",\"currency\":\"IDR\",\"currencyRange\":[\"IDR\"],\"maxValue\":\"\",\"minValue\":\"\"},\"value\":0},{\"id\":\"widget16987239118090001\",\"name\":\"PPH Amount\",\"type\":\"amount\",\"ext\":{\"capitalValue\":\"贰仟万元整\",\"currency\":\"IDR\",\"currencyRange\":[\"IDR\"],\"maxValue\":\"\",\"minValue\":\"\"},\"value\":20000000},{\"id\":\"widget16987480260760001\",\"name\":\"Nama Rekening账户名称 \",\"type\":\"radioV2\",\"ext\":null,\"value\":\"WILSON YOHAN\",\"option\":{\"key\":\"id_V0lMU09OJTIwWU9IQU4=\",\"text\":\"WILSON YOHAN\"}},{\"id\":\"widget16987481753790001\",\"name\":\" No.Bank Rekening账号\",\"type\":\"radioV2\",\"ext\":null,\"value\":\"1060009880777\",\"option\":{\"key\":\"id_MTA2MDAwOTg4MDc3Nw==\",\"text\":\"1060009880777\"}},{\"id\":\"widget16987489371320001\",\"name\":\"Nama Bank银行名称 \",\"type\":\"radioV2\",\"ext\":null,\"value\":\"MANDIRI\",\"option\":{\"key\":\"id_TUFORElSSQ==\",\"text\":\"MANDIRI\"}},{\"id\":\"widget16987228202520001\",\"name\":\"流水号\",\"type\":\"serialNumber\",\"ext\":null,\"value\":\"PR20240525013814\"},{\"id\":\"widget16510510447300001\",\"name\":\"附件\",\"type\":\"attachmentV2\",\"ext\":\"Contract Application 合同审批.pdf,JHON ENROE.jpg,Perjanjian Sewa Menyewa (John Enroe P) PT Global Jet Cargo.pdf,SHGB 4538 (Blok 99-R).pdf,003WP-INVV2024.pdf\",\"value\":[\"https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MWVlY2JhOGI0MjVmNmQwOThiNThkZmZjYjZiMDAwMGNfMGFkMTdkNjFiNWUyZmVlNjFmZGQ4NTY0MGE3YWFmMzNfSUQ6NzM3MTY0NDQ2MzAwMjgxMjQ0OF8xNzE2NjQ5MjY0OjE3MTY3MzU2NjRfVjM\",\"https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWYzMzU3ZDVmNzdhYjM5ZTY1NjkzMTk0NDI5N2FlZGVfZDc2Nzg3M2JiZDE2ZjkxN2NjZGE4YzU2MzE0Zjc2NThfSUQ6NzM3MTY0NDQ2MjU1NDMxNjgzMl8xNzE2NjQ5MjY0OjE3MTY3MzU2NjRfVjM\",\"https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzM0NTRmYzExMWRlYzhhYTYzODE1YmU3ZmE2YzY3ODlfZGIyMTIyMGU5ZDIzMzQ4OTM1ZmRiOTQwZTEzODhkYTBfSUQ6NzM3MTY0NDQ2MTA1NjczNzMxMV8xNzE2NjQ5MjY0OjE3MTY3MzU2NjRfVjM\",\"https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MGRlZWZmNmI1Y2Y3NjgxZWFlZmUzNDg3MDVmMTcwNWVfMzFjMzI1MjQxODc1YWFkMzkzNGRiYTdkYjAyMWY3MGFfSUQ6NzM3MTY0NDQ2MDczMzcyNjc1Ml8xNzE2NjQ5MjY0OjE3MTY3MzU2NjRfVjM\",\"https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NmRkOTdkZDc3NWFlZWZkMmYxMGMzNjFlM2FmZjA4MGFfMjMwMjZhODliY2I1ZGQxYzdmMDlhNWQzMTJkMTg3YjhfSUQ6NzM3Mjg0ODMxODAwNDcwNzM2MF8xNzE2NjQ5MjY1OjE3MTY3MzU2NjVfVjM\"]},{\"id\":\"widget16987231311320001\",\"name\":\"说明 1\",\"type\":\"text\",\"ext\":null,\"value\":\"Attachment - \\nLampirkan semua lampiran ( payment form, invoice, kontrak kerjasama, faktur pajak, payment form, foto renovasi DSO,foto FA,quotation,issue approval ,etc).\\n\\nSemua lampiran harus dipindai secara vertikal, bukan horizontal.\\n\\nUtk pembayaran rute external dan internal, harus lampirkan \\\"Tabel Rincian Ledger Pengiriman\\\" - 附件-\\n附上所有附件（付款单、发票、合作合同、税务发票、付款单、DSO装修照片、FA照片、报价单、签发审批等）。\\n\\n所有附件应垂直扫描，而不是水平扫描。\\n\\n对于外部和内部途径付款，您必须附上“交货分类明细表”\"}]";
		List<Widget> widgets = Jsons.DEFAULT.fromJson(json, new TypeToken<List<Widget>>() {}.getType());
		for(Widget widget : widgets) {
			if (widget.getValue() instanceof ArrayList) {
				List<String> list = (List<String>) widget.getValue();
				System.out.println(list.size());
			}
//			System.out.printf(widget.getName());
//			System.out.printf("              ");
//			System.out.printf(widget.getType());
//			System.out.printf("              ");
////			System.out.printf((String) widget.getValue());
//			System.out.println(widget.getValue().getClass());

		}
	}

}


