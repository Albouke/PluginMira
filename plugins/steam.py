import re
import os
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime
import requests
from enum import Enum
from Lib import *
from Lib.core import PluginManager, ConfigManager

logger = Logger.get_logger()

plugin_info = PluginManager.PluginInfo(
  NAME="steamWatcher",
  AUTHOR="MMG",
  VERSION="1.0.0",
  DESCRIPTION="用于反馈格式化steam信息的插件",
  HELP_MSG="/视奸 or /watch 启用，附加#qq#，#steam好友代码# 或 #steam 64id#查询，无参数情况下默认查询自身，使用/bind or /绑定 + #steam好友代码# 或 #steam 64id# 创建自己的查询关系"
)
watch = EventHandlers.CommandRule("watch", aliases={"视奸"})
bind = EventHandlers.CommandRule("bind", aliases={"绑定"})

watchmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[watch])
bindmatcher = EventHandlers.on_event(EventClassifier.GroupMessageEvent, priority=0, rules=[bind])


class PlayerState(Enum):
  OFFLINE = 0
  ONLINE = 1
  BUSY = 2
  AWAY = 3
  SNOOZE = 4
  LOOKING_TO_TRADE = 5
  LOOKING_TO_PLAY = 6
  IN_GAME = 7


class SteamIDConverter:
  """Steam ID 转换工具类"""
  # Steam 64位ID基准值
  STEAM64_BASE = 76561197960265728

  @staticmethod
  def is_steam_64id(id_str: str) -> bool:
    """检查是否为有效的Steam 64位ID"""
    return (id_str.isdigit() and
            len(id_str) == 17 and
            id_str.startswith('7656'))

  @staticmethod
  def is_friend_code(code: str) -> bool:
    """检查是否为Steam好友代码格式"""
    # Steam好友代码通常是比较短的数字
    return code.isdigit() and 7 <= len(code) <= 10

  @staticmethod
  def is_qq_id(id_str: str) -> bool:
    """检查是否为QQ号码格式"""
    # QQ号一般是5到11位的数字
    return id_str.isdigit() and 5 <= len(id_str) <= 11

  @staticmethod
  def friend_code_to_steam_64id(friend_code: str) -> str:
    """将Steam好友代码转换为64位ID"""
    # 好友代码是Steam 64位ID减去基准值
    try:
      friend_code_int = int(friend_code)
      steam_64id = friend_code_int + SteamIDConverter.STEAM64_BASE
      return str(steam_64id)
    except ValueError:
      return None

  @staticmethod
  def steam_64id_to_friend_code(steam_64id: str) -> str:
    """将64位ID转换为Steam好友代码"""
    try:
      steam_id_int = int(steam_64id)
      friend_code = steam_id_int - SteamIDConverter.STEAM64_BASE
      return str(friend_code)
    except ValueError:
      return None


class SteamAPI:
  def __init__(self, api_key: str):
    self.api_key = api_key
    self.base_url = "https://api.steampowered.com"

  def format_player_info(self, steam_id: str) -> str:
    """格式化玩家所有信息并返回完整字符串"""
    result = []

    # 获取基本信息
    player_data = self._make_request("/ISteamUser/GetPlayerSummaries/v2/", {
      'key': self.api_key,
      'steamids': steam_id
    })

    if not player_data or 'response' not in player_data or not player_data['response']['players']:
      return "无法获取玩家信息，该账号可能不存在或设为私密"

    player = player_data['response']['players'][0]
    result.append(self._format_basic_info(player))

    # 获取当前游戏信息
    current_game = self.get_current_game(steam_id)
    if current_game:
      result.append(self._format_current_game(current_game))

    # 获取最近游戏
    recent_games = self._make_request("/IPlayerService/GetRecentlyPlayedGames/v1/", {
      'key': self.api_key,
      'steamid': steam_id,
      'count': 10
    })
    if recent_games:
      result.append(self._format_recent_games(recent_games))

    # 获取游戏库信息
    owned_games = self._make_request("/IPlayerService/GetOwnedGames/v1/", {
      'key': self.api_key,
      'steamid': steam_id,
      'include_appinfo': 1,
      'include_played_free_games': 1
    })
    if owned_games:
      result.append(self._format_owned_games(owned_games))

    return "\n".join(result)

  def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
    """统一的API请求处理方法"""
    url = self.base_url + endpoint
    try:
      response = requests.get(url, params=params)
      response.raise_for_status()
      return response.json()
    except requests.exceptions.RequestException as e:
      print(f"API请求错误: {e}")
      return None

  def _get_players_summaries(self, steam_ids: List[str]) -> Optional[Dict]:
    """获取玩家详细信息"""
    endpoint = "/ISteamUser/GetPlayerSummaries/v2/"
    params = {
      'key': self.api_key,
      'steamids': ','.join(steam_ids)
    }
    data = self._make_request(endpoint, params)
    if not data:
      return None

    players = {}
    for player in data.get('response', {}).get('players', []):
      players[player['steamid']] = {
        'steam_id': player['steamid'],
        'name': player.get('personaname', 'Unknown'),
        'profile_url': player.get('profileurl', ''),
        'avatar': player.get('avatarfull', ''),
        'status': player.get('personastate', 0),
        'visibility': player.get('communityvisibilitystate', 1),
        'last_online': player.get('lastlogoff', 0),
        'real_name': player.get('realname', ''),
        'country': player.get('loccountrycode', ''),
        'current_game': {
          'id': player.get('gameid', ''),
          'name': player.get('gameextrainfo', '')
        } if 'gameid' in player else None
      }
    return players

  def get_friend_list_with_details(self, steam_id: str) -> Optional[List[Dict]]:
    """获取好友列表及其详细信息"""
    friends_data = self._make_request("/ISteamUser/GetFriendList/v1/", {
      'key': self.api_key,
      'steamid': steam_id,
      'relationship': 'friend'
    })

    if not friends_data:
      return None

    friends = friends_data.get('friendslist', {}).get('friends', [])
    friend_ids = [friend['steamid'] for friend in friends]
    friend_details = self._get_players_summaries(friend_ids)

    if not friend_details:
      return None

    return [{
      **friend_details[friend['steamid']],
      'friendship_since': friend.get('friend_since', 0)
    } for friend in friends if friend['steamid'] in friend_details]

  def get_current_game(self, steam_id: str) -> Optional[Dict]:
    """获取玩家当前游戏状态"""
    player_data = self._make_request("/ISteamUser/GetPlayerSummaries/v2/", {
      'key': self.api_key,
      'steamids': steam_id
    })

    if not player_data or 'response' not in player_data or not player_data['response']['players']:
      return None

    player = player_data['response']['players'][0]
    player_state = PlayerState(player.get('personastate', 0))

    result = {
      'name': player.get('personaname', 'Unknown'),
      'state': player_state.name,
      'game_info': None
    }

    if 'gameid' in player:
      result['game_info'] = {
        'game_id': player['gameid'],
        'game_name': player.get('gameextrainfo', 'Unknown Game'),
        'server_ip': player.get('gameserverip', None),
        'server_steam_id': player.get('gameserversteamid', None)
      }

    return result

  def _format_basic_info(self, player: Dict) -> str:
    """格式化基本信息"""
    lines = ["==== 玩家基本信息 ===="]
    lines.append(f"名称: {player.get('personaname', 'Unknown')}")
    lines.append(f"Steam ID: {player.get('steamid', 'N/A')}")
    lines.append(f"好友代码: {SteamIDConverter.steam_64id_to_friend_code(player.get('steamid', '0'))}")
    lines.append(f"个人链接: {player.get('profileurl', 'N/A')}")
    lines.append(f"国家: {player.get('loccountrycode', 'N/A')}")

    last_logoff = player.get('lastlogoff', 0)
    if last_logoff:
      last_online = datetime.fromtimestamp(last_logoff).strftime('%Y-%m-%d %H:%M:%S')
      lines.append(f"上次在线: {last_online}")

    return "\n".join(lines)

  def _format_current_game(self, current_game: Dict) -> str:
    """格式化当前游戏状态"""
    lines = ["==== 当前状态 ===="]
    lines.append(f"状态: {current_game['state']}")
    if current_game['game_info']:
      lines.append(f"正在游玩: {current_game['game_info']['game_name']}")

    return "\n".join(lines)

  def _format_recent_games(self, recent_games: Dict) -> str:
    """格式化最近游戏信息"""
    lines = []
    if 'response' in recent_games and 'games' in recent_games['response']:
      lines.append("==== 最近玩过的游戏 ====")
      for game in recent_games['response']['games'][:3]:  # 只显示前3个游戏
        playtime = game.get('playtime_forever', 0)
        hours = playtime // 60
        minutes = playtime % 60
        lines.append(f"游戏: {game.get('name', 'Unknown')}")
        lines.append(f"总计游玩时间: {hours}小时{minutes}分钟")
        if 'playtime_2weeks' in game:
          recent_playtime = game['playtime_2weeks']
          recent_hours = recent_playtime // 60
          recent_minutes = recent_playtime % 60
          lines.append(f"最近两周游玩时间: {recent_hours}小时{recent_minutes}分钟")
        lines.append("-" * 30)

    return "\n".join(lines)

  def _format_owned_games(self, owned_games: Dict) -> str:
    """格式化游戏库信息"""
    lines = ["==== 游戏库统计 ===="]

    if 'response' in owned_games and 'games' in owned_games['response']:
      games_count = len(owned_games['response']['games'])
      lines.append(f"拥有游戏数量: {games_count}")

      sorted_games = sorted(
        owned_games['response']['games'],
        key=lambda x: x.get('playtime_forever', 0),
        reverse=True
      )[:5]

      lines.append("\n最常玩的5个游戏:")
      for game in sorted_games:
        playtime = game.get('playtime_forever', 0)
        hours = playtime // 60
        minutes = playtime % 60
        lines.append(f"游戏: {game.get('name', 'Unknown')}")
        lines.append(f"总计游玩时间: {hours}小时{minutes}分钟")
        lines.append("-" * 30)

    return "\n".join(lines)

  def resolve_vanity_url(self, vanity_url: str) -> Optional[str]:
    """将Steam自定义URL解析为64位ID"""
    data = self._make_request("/ISteamUser/ResolveVanityURL/v1/", {
      'key': self.api_key,
      'vanityurl': vanity_url
    })

    if data and data.get('response', {}).get('success') == 1:
      return data['response']['steamid']
    return None


class SteamRelation:
  def __init__(self):
    # 使用相对路径存储数据库文件
    db_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(db_dir, 'Relation.db')

    try:
      self.conn = sqlite3.connect(db_path)
      self.cursor = self.conn.cursor()

      # 创建表
      self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS steamRelation (
                    qqID TEXT PRIMARY KEY,
                    steamID TEXT NOT NULL
                )
            ''')
      self.conn.commit()
    except sqlite3.Error as e:
      print(f"数据库连接错误: {e}")
      # 确保在错误时也有conn属性，防止__del__报错
      self.conn = None

  def insert_relation(self, qq_id: str, steam_id: str):
    """插入QQ和Steam关联记录"""
    if not self.conn:
      return False

    try:
      # 先检查是否已存在记录，如果存在则更新
      self.cursor.execute('''
        SELECT * FROM steamRelation WHERE qqID = ?
      ''', (qq_id,))

      if self.cursor.fetchone():
        # 更新已有记录
        self.cursor.execute('''
          UPDATE steamRelation SET steamID = ? WHERE qqID = ?
        ''', (steam_id, qq_id))
      else:
        # 插入新记录
        self.cursor.execute('''
          INSERT INTO steamRelation (qqID, steamID) VALUES (?, ?)
        ''', (qq_id, steam_id))

      self.conn.commit()
      return True
    except sqlite3.Error:
      return False

  def get_steam_id(self, qq_id: str) -> str:
    """根据QQ ID查询Steam ID"""
    if not self.conn:
      return None

    try:
      self.cursor.execute('''
                SELECT steamID FROM steamRelation
                WHERE qqID = ?
            ''', (qq_id,))
      result = self.cursor.fetchone()
      return result[0] if result else None
    except sqlite3.Error:
      return None

  def check_steam_id(self, steam_id: str) -> str:
    """
    查询Steam ID是否已被使用
    返回: 关联的QQ ID，未关联则返回None
    """
    if not self.conn:
      return None

    try:
      self.cursor.execute('''
              SELECT qqID FROM steamRelation
              WHERE steamID = ?
          ''', (steam_id,))
      result = self.cursor.fetchone()
      return result[0] if result else None
    except sqlite3.Error:
      return None

  def __del__(self):
    """析构函数,关闭数据库连接"""
    if hasattr(self, 'conn') and self.conn:
      self.conn.close()


def process_steam_id_input(input_id, api_key):
  """
  处理输入的ID，判断类型并返回标准化的Steam 64位ID
  """
  steam_api = SteamAPI(api_key)

  # 检查是否为Steam 64位ID
  if SteamIDConverter.is_steam_64id(input_id):
    return input_id

  # 检查是否为Steam好友代码
  elif SteamIDConverter.is_friend_code(input_id):
    return SteamIDConverter.friend_code_to_steam_64id(input_id)

  # 尝试作为自定义URL解析
  else:
    vanity_id = steam_api.resolve_vanity_url(input_id)
    if vanity_id:
      return vanity_id

  # 如果都不匹配，返回原输入，让API尝试处理
  return input_id


@bindmatcher.register_handler()
def binder(event_data):
  # 从原始消息中提取命令和ID部分
  msg = event_data.raw_message

  # 添加调试信息
  print(f"Debug - Raw message: '{msg}'")

  # 修改正则表达式，使用更宽松的匹配方式
  match = re.search(r'(bind|绑定).*?#(.*?)#', msg, re.IGNORECASE)

  if not match:
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        "请使用正确的格式: /bind #steamID# 或 /绑定 #steamID#"
      ),
      group_id=event_data.group_id
    ).call()
    return

  input_id = match.group(2).strip()
  qq_id = str(event_data.user_id)

  if not input_id:
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        "请提供有效的Steam ID、好友代码或自定义URL"
      ),
      group_id=event_data.group_id
    ).call()
    return

  API_KEY = "E346980886DC7CA15CC9B5E87A4109B4"
  relation = SteamRelation()

  # 处理输入ID
  steam_id = process_steam_id_input(input_id, API_KEY)

  if not steam_id:
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        "无法识别输入的ID格式，请提供有效的Steam ID或好友代码"
      ),
      group_id=event_data.group_id
    ).call()
    return

  # 检查Steam ID是否已被其他QQ绑定
  bound_qq = relation.check_steam_id(steam_id)
  if bound_qq and bound_qq != qq_id:
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        f"该Steam ID已被QQ {bound_qq} 绑定"
      ),
      group_id=event_data.group_id
    ).call()
    return

  # 绑定关系
  if relation.insert_relation(qq_id, steam_id):
    friend_code = SteamIDConverter.steam_64id_to_friend_code(steam_id)
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        f"绑定成功！\nQQ: {qq_id}\nSteam ID: {steam_id}\n好友代码: {friend_code}"
      ),
      group_id=event_data.group_id
    ).call()
  else:
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        "绑定失败，请稍后重试"
      ),
      group_id=event_data.group_id
    ).call()


@watchmatcher.register_handler()
def watcher(event_data):
  # 从原始消息中提取命令和ID部分
  msg = event_data.raw_message

  API_KEY = "E346980886DC7CA15CC9B5E87A4109B4"
  relation = SteamRelation()
  steam_api = SteamAPI(API_KEY)

  # 匹配 watch #ID# 或 视奸 #ID# 格式
  match = re.search(r'(?:/?)(?:watch|视奸)\s*#([^#]+)#', msg, re.IGNORECASE)

  # 确定要查询的steam_id
  if not match:
    # 无参数查询，查询发送者自己绑定的ID
    qq_id = str(event_data.user_id)
    steam_id = relation.get_steam_id(qq_id)

    if not steam_id:
      Actions.SendMsg(
        message=QQRichText.QQRichText(
          f"您尚未绑定Steam ID。请使用 /bind #steamID# 或 /绑定 #steamID# 绑定您的Steam账号"
        ),
        group_id=event_data.group_id
      ).call()
      return
  else:
    # 有参数查询
    input_id = match.group(1).strip()
    steam_id = None

    # 检查输入类型并确定steam_id，按优先级尝试各种解释
    if SteamIDConverter.is_steam_64id(input_id):
      # 是Steam 64位ID，直接使用
      steam_id = input_id
    elif SteamIDConverter.is_qq_id(input_id):
      # 是QQ号，查询数据库关系
      steam_id = relation.get_steam_id(input_id)

      # 如果没有找到QQ对应的绑定关系，且符合好友代码格式，尝试作为好友代码转换
      if not steam_id and SteamIDConverter.is_friend_code(input_id):
        steam_id = SteamIDConverter.friend_code_to_steam_64id(input_id)
    elif SteamIDConverter.is_friend_code(input_id):
      # 是好友代码，转换为64id
      steam_id = SteamIDConverter.friend_code_to_steam_64id(input_id)

    # 如果以上都不成功，尝试作为自定义URL解析
    if not steam_id:
      steam_id = steam_api.resolve_vanity_url(input_id)

    # 如果所有尝试都失败，返回错误信息
    if not steam_id:
      Actions.SendMsg(
        message=QQRichText.QQRichText(
          f"无法解析输入 '{input_id}'，可能原因：\n"
          f"1. 如果您输入的是QQ号，该QQ未绑定Steam账号\n"
          f"2. 如果您输入的是好友代码，格式可能不正确\n"
          f"3. 如果您输入的是自定义URL，可能不存在\n\n"
          f"请提供有效的Steam ID、已绑定的QQ号或正确的好友代码"
        ),
        group_id=event_data.group_id
      ).call()
      return

  # 获取Steam玩家信息
  try:
    player_info = steam_api.format_player_info(steam_id)

    if player_info:
      # 发送查询结果
      Actions.SendMsg(
        message=QQRichText.QQRichText(
          player_info
        ),
        group_id=event_data.group_id
      ).call()
    else:
      Actions.SendMsg(
        message=QQRichText.QQRichText(
          "获取玩家信息失败，该账号可能设为私密或不存在"
        ),
        group_id=event_data.group_id
      ).call()
  except Exception as e:
    Actions.SendMsg(
      message=QQRichText.QQRichText(
        f"查询出错: {str(e)}"
      ),
      group_id=event_data.group_id
    ).call()
